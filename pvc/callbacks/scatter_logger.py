# pvc/callbacks/scatter_logger.py

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from pvc.plots import scatter

class ScatterLogger(Callback):
    def __init__(self, every_n_val_epochs=1, units="s",
                 max_points=8000, clip_quantile=0.999):
        self.n = every_n_val_epochs
        self.units = units
        self.max_points = max_points
        self.clip = clip_quantile

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.n != 0:
            return
        y, yhat = pl_module._gather_epoch_preds("val")
        y = y.detach().cpu().numpy()
        yhat = yhat.detach().cpu().numpy()
        fig = scatter(y, yhat,
                                  title=f"Validation — epoch {trainer.current_epoch}",
                                  units=self.units,
                                  max_points=self.max_points,
                                  clip_quantile=self.clip)
        trainer.logger.log_image(
            key="val/scatter", images=[fig], caption=[f"epoch {trainer.current_epoch}"],
            step=trainer.current_epoch
        )
        plt.close(fig)

    def on_test_epoch_end(self, trainer, pl_module):
        y, yhat = pl_module._gather_epoch_preds("test")
        y = y.detach().cpu().numpy()
        yhat = yhat.detach().cpu().numpy()
        fig = scatter(y, yhat,
                                  title="Test set", units=self.units,
                                  max_points=self.max_points, clip_quantile=self.clip)
        trainer.logger.log_image(
            key="test/scatter", images=[fig], caption=["test set"],
            step=trainer.current_epoch
        )
        plt.close(fig)
