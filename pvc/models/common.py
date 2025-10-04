# pig_pvc/models/common.py
import lightning as L
import torch
import torch.nn as nn


# ---------- building blocks ----------
def conv_bn_relu(cin, cout, k, stride=1, padding=None, groups=1):
    if padding is None:
        padding = (k - 1) // 2
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, stride, padding, groups=groups, bias=False),
        nn.BatchNorm1d(cout),
        nn.ReLU(inplace=True),
    )


class PredCacheMixin(L.LightningModule):
    """
    Adds self._cache = {"val": {"y": [], "yhat": []},
                        "test": {"y": [], "yhat": []}}
    and convenience hooks to every LightningModule.
    """

    def on_validation_epoch_start(self) -> None:
        self._cache = getattr(self, "_cache", {})
        self._cache["val"] = {"y": [], "yhat": []}

    def on_test_epoch_start(self) -> None:
        self._cache = getattr(self, "_cache", {})
        self._cache["test"] = {"y": [], "yhat": []}

    # call this in your validation_step / test_step
    def _stash_preds(self, y, yhat, stage: str):
        self._cache[stage]["y"].append(y.detach().cpu())
        self._cache[stage]["yhat"].append(yhat.detach().cpu())

    def _gather_epoch_preds(self, stage: str):
        ys = torch.cat(self._cache[stage]["y"], dim=0)
        yhats = torch.cat(self._cache[stage]["yhat"], dim=0)
        return ys, yhats


class LeadMixer(nn.Module):
    """1x1 conv that maps 247 leads → base_channels."""

    def __init__(self, base_channels=64):
        super().__init__()
        self.mix = conv_bn_relu(247, base_channels, k=1, padding=0)

    def forward(self, x):  # x: [B,247,430]
        return self.mix(x)  # → [B,base_channels,430]


class RegressorHead(nn.Module):
    """Two-head heteroscedastic regression."""

    def __init__(self, fin):
        super().__init__()
        self.mu = nn.Linear(fin, 1)
        self.logvar = nn.Linear(fin, 1)

    def forward(self, z):
        return self.mu(z).squeeze(-1), self.logvar(z).squeeze(-1)

class ClassifierHead(nn.Module):
    """
    A simple classification head that outputs a single logit per sample.
    Mirrors the style of RegressorHead: small, explicit, no surprises.

    Args
    ----
    in_dim : int
        Dimension of the input feature vector (backbone output).
    p_drop : float
        Optional dropout before the final linear layer.
    """
    def __init__(self, in_dim: int, p_drop: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p_drop) if p_drop and p_drop > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : [B, in_dim]
            Backbone feature vectors

        Returns
        -------
        logit : [B]
            Raw (pre-sigmoid) scores. Apply torch.sigmoid(logit) for probabilities.
        """
        return self.fc(self.dropout(z)).squeeze(-1)
    
def hetero_gaussian_nll(mu, logvar, y):
    inv_var = torch.exp(-logvar)
    return 0.5 * ((y - mu) ** 2 * inv_var + logvar).mean()


def huber_loss(pred, target, delta: float):
    """Wrapper around PyTorch's built-in SmoothL1 (β = δ)."""
    criterion = nn.SmoothL1Loss(beta=delta)
    return criterion(pred, target)


def masked_mean_time(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """
    x: [B, C, L], mask: [B, L] (True=valid). Returns [B, C].
    """
    if mask is None:
        return x.mean(dim=-1)
    m = mask[:, None, :].to(dtype=x.dtype)        # [B,1,L]
    s = (x * m).sum(dim=-1)
    z = s / (m.sum(dim=-1).clamp_min(1e-6))
    return z
    
# ---------- master LightningModule ----------
class BasePVCModel(PredCacheMixin, L.LightningModule):
    def __init__(
        self, backbone: nn.Module, lr: float = 3e-4, loss_type="nll", huber_delta=10.0, T_max=50, task="regression", cls_window_s=180
    ):
        super().__init__()
        self.backbone = backbone  # out: [B,F]
        fin = backbone.out_channels  # every backbone sets this
        self.task = task
        self.head = RegressorHead(fin) if task == "regression" else ClassifierHead(fin)
        self.T_max = T_max
        self.cls_window_s = cls_window_s
        self.save_hyperparameters(ignore=["backbone"])

    def _unpack(self, batch):
        # works for dicts from our collate
        if isinstance(batch, dict):
            return batch["potvals"], batch["time_to_pvc"], batch.get("mask", None)
        # fallback (tuple)
        x, y = batch
        return x, y, None

    # -- forward -------------------------------------------------------------
    def forward(self, x, mask=None):
        # forward signatures of backbones will accept mask if provided
        if "mask" in self.backbone.forward.__code__.co_varnames:
            z = self.backbone(x, mask=mask)
        else:
            # legacy backbones that already pool internally (unmasked)
            z = self.backbone(x)
        return self.head(z)

    # -- shared training / validation / test --------------------------------
    def step(self, batch, stage: str, task="regression"):
        x, y, mask = self._unpack(batch)
        if task == "regression":
            mu, logv = self(x, mask=mask)

            if self.hparams.loss_type == "huber":
                loss = huber_loss(mu, y, delta=self.hparams.huber_delta)
            else:  # default = heteroscedastic NLL
                logv = logv.clamp(-6, 6)  # inside BasePVCModel.step before NLL
                loss = hetero_gaussian_nll(mu, logv, y)

            self.log(f"{stage}/loss", loss, prog_bar=(stage == "val"))
            if stage in ("val", "test"):
                mae = (mu - y).abs().mean()
                self.log(f"{stage}/mae", mae, prog_bar=True)
                self._stash_preds(y, mu, stage)
            return loss

        elif task == "classification":
            y = (y <= self.cls_window_s).float()
            yhat = self(x, mask=mask)
            loss = nn.BCEWithLogitsLoss()(yhat, y)
            self.log(f"{stage}/loss", loss, prog_bar=(stage == "val"))
            if stage in ("val", "test"):
                acc = ((yhat.sigmoid() > 0.5) == y).float().mean()
                self.log(f"{stage}/acc", acc, prog_bar=True)
                self._stash_preds(y, yhat, stage)
            return loss

    def training_step(self, batch, _):
        return self.step(batch, "train", task=self.hparams.task)

    def validation_step(self, batch, _):
        return self.step(batch, "val", task=self.hparams.task)

    def test_step(self, batch, _):
        return self.step(batch, "test", task=self.hparams.task)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.T_max)
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val/loss"}
