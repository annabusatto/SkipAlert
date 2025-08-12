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


def hetero_gaussian_nll(mu, logvar, y):
    inv_var = torch.exp(-logvar)
    return 0.5 * ((y - mu) ** 2 * inv_var + logvar).mean()


def huber_loss(pred, target, delta: float):
    """Wrapper around PyTorch's built-in SmoothL1 (β = δ)."""
    criterion = nn.SmoothL1Loss(beta=delta)
    return criterion(pred, target)


# ---------- master LightningModule ----------
class BasePVCModel(PredCacheMixin, L.LightningModule):
    def __init__(
        self, backbone: nn.Module, lr: float = 3e-4, loss_type="nll", huber_delta=10.0, T_max=50
    ):
        super().__init__()
        self.backbone = backbone  # out: [B,F]
        fin = backbone.out_channels  # every backbone sets this
        self.head = RegressorHead(fin)
        self.T_max = T_max
        self.save_hyperparameters(ignore=["backbone"])

    # -- forward -------------------------------------------------------------
    def forward(self, x):
        z = self.backbone(x)
        return self.head(z)

    # -- shared training / validation / test --------------------------------
    def step(self, batch, stage: str):
        x, y = batch
        mu, logv = self(x)

        if self.hparams.loss_type == "huber":
            loss = huber_loss(mu, y, delta=self.hparams.huber_delta)
        else:  # default = heteroscedastic NLL
            logv = logv.clamp(-6, 6)  # inside BasePVCModel.step before NLL
            loss = hetero_gaussian_nll(mu, logv, y)

        self.log(f"{stage}_loss", loss, prog_bar=(stage == "val"))
        if stage in ("val", "test"):
            mae = (mu - y).abs().mean()
            self.log(f"{stage}_mae", mae, prog_bar=True)
            self._stash_preds(y, mu, stage)
        return loss

    def training_step(self, batch, _):
        return self.step(batch, "train")

    def validation_step(self, batch, _):
        return self.step(batch, "val")

    def test_step(self, batch, _):
        return self.step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.T_max)
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "val_loss"}
