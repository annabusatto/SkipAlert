# pvc/callbacks/confmat_logger.py
import math
import torch
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback

_EPS = 1e-12


def _counts(pred: torch.Tensor, labels: torch.Tensor):
    """Return (tp, fp, fn, tn) as torch.Tensors of shape []."""
    tp = (pred.eq(1) & labels.eq(1)).sum()
    fp = (pred.eq(1) & labels.eq(0)).sum()
    fn = (pred.eq(0) & labels.eq(1)).sum()
    tn = (pred.eq(0) & labels.eq(0)).sum()
    return tp, fp, fn, tn


def _prf(tp, fp, fn):
    precision = tp.float() / (tp + fp + _EPS)
    recall    = tp.float() / (tp + fn + _EPS)
    f1        = 2 * precision * recall / (precision + recall + _EPS)
    return precision, recall, f1


def _youden_best_thresh(probs: torch.Tensor, labels: torch.Tensor, steps: int = 101) -> float:
    """Pick threshold that maximizes Youden's J = TPR - FPR."""
    grid = torch.linspace(0, 1, steps=steps)
    best_j, best_t = -1.0, 0.5
    with torch.no_grad():
        for t in grid:
            pred = (probs >= t).int()
            tp, fp, fn, tn = _counts(pred, labels)
            tpr = tp.item() / max(1, (tp + fn).item())
            fpr = fp.item() / max(1, (fp + tn).item())
            j = tpr - fpr
            if j > best_j:
                best_j, best_t = j, float(t)
    return best_t


def _confmat_figure(tp, fp, fn, tn, title: str, normalize: bool, thresh: float | None):
    """Return a matplotlib Figure of a 2x2 confusion matrix."""
    # import numpy as np
    cm = torch.stack([torch.stack([tn, fp]), torch.stack([fn, tp])]).float()  # [[TN,FP],[FN,TP]]
    if normalize:
        row_sums = cm.sum(dim=1, keepdim=True).clamp_min(_EPS)
        cm_plot = (cm / row_sums).numpy()
        fmt = "{:.2f}"
    else:
        cm_plot = cm.numpy()
        fmt = "{}"

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm_plot, cmap="Blues", vmin=0.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["No PVC ≤ win", "PVC ≤ win"],
           yticklabels=["No PVC ≤ win", "PVC ≤ win"],
           ylabel="True", xlabel="Predicted")
    for i in range(2):
        for j in range(2):
            val = cm_plot[i, j]
            ax.text(j, i, fmt.format(int(val) if not normalize else val),
                    ha="center", va="center",
                    color="white" if val > (cm_plot.max() / 2) else "black", fontsize=11)
    hdr = title
    if thresh is not None:
        hdr += f" (thr={thresh:.2f})"
    ax.set_title(hdr)
    fig.tight_layout()
    return fig


class ConfusionMatrixLogger(Callback):
    """
    Logs a confusion matrix image + precision/recall/F1 for classification runs.

    Assumes pl_module._stash_preds(label, logit, stage) was called during {val,test}_step.
    Works only when pl_module.hparams.task == "classification".
    """

    def __init__(
        self,
        every_n_val_epochs: int = 1,
        threshold: float | None = None,     # if None, uses best val threshold (Youden J)
        derive_threshold: bool = True,      # compute Youden J on each val epoch if threshold is None
        normalize: bool = False,            # show rates instead of counts
    ):
        super().__init__()
        self.n = int(every_n_val_epochs)
        self.fixed_thresh = threshold
        self.derive = bool(derive_threshold)
        self.normalize = bool(normalize)

    # ────────── helpers ──────────
    def _safe_gather(self, trainer, pl_module, stage: str):
        if not hasattr(pl_module, "_gather_epoch_preds"):
            return None, None
        try:
            y, yhat = pl_module._gather_epoch_preds(stage)  # labels, logits
            return y, yhat
        except Exception:
            return None, None

    # ────────── Validation ──────────
    def on_validation_epoch_end(self, trainer, pl_module):
        # only for classification tasks
        if getattr(pl_module.hparams, "task", "regression") != "classification":
            return
        if trainer.current_epoch % self.n != 0:
            return

        labels, logits = self._safe_gather(trainer, pl_module, "val")
        if labels is None or logits is None or labels.numel() == 0:
            return

        labels = labels.int()
        probs = logits.sigmoid()

        # pick threshold
        if self.fixed_thresh is not None:
            thr = float(self.fixed_thresh)
        elif self.derive:
            thr = _youden_best_thresh(probs, labels, steps=101)
            # stash on module so test can reuse it
            setattr(pl_module, "best_thresh", thr)
        else:
            thr = getattr(pl_module, "best_thresh", 0.5)

        pred = (probs >= thr).int()
        tp, fp, fn, tn = _counts(pred, labels)
        precision, recall, f1 = _prf(tp, fp, fn)

        # log scalars
        pl_module.log("val/precision", precision, prog_bar=True)
        pl_module.log("val/recall",    recall,    prog_bar=True)
        pl_module.log("val/f1",        f1,        prog_bar=True)

        # image
        fig = _confmat_figure(tp, fp, fn, tn,
                              title=f"Validation — epoch {trainer.current_epoch}",
                              normalize=self.normalize, thresh=thr)
        trainer.logger.log_image(
            key="val/confusion_matrix",
            images=[fig],
            caption=[f"epoch {trainer.current_epoch}"],
            step=trainer.current_epoch,
        )
        plt.close(fig)

    # ────────── Test ──────────
    def on_test_epoch_end(self, trainer, pl_module):
        if getattr(pl_module.hparams, "task", "regression") != "classification":
            return

        labels, logits = self._safe_gather(trainer, pl_module, "test")
        if labels is None or logits is None or labels.numel() == 0:
            return

        labels = labels.int()
        probs = logits.sigmoid()

        thr = (float(self.fixed_thresh) if self.fixed_thresh is not None
               else getattr(pl_module, "best_thresh", 0.5))
        pred = (probs >= thr).int()
        tp, fp, fn, tn = _counts(pred, labels)
        precision, recall, f1 = _prf(tp, fp, fn)

        pl_module.log("test/precision", precision, prog_bar=True)
        pl_module.log("test/recall",    recall,    prog_bar=True)
        pl_module.log("test/f1",        f1,        prog_bar=True)

        fig = _confmat_figure(tp, fp, fn, tn,
                              title="Test set",
                              normalize=self.normalize, thresh=thr)
        trainer.logger.log_image(
            key="test/confusion_matrix",
            images=[fig],
            caption=["test set"],
            step=trainer.current_epoch,
        )
        plt.close(fig)
