# pvc/models/pvc_module.py
from .common import BasePVCModel
from .backbones.sequence_wrapper import SequenceWrapper
from . import factory                     # registry (see below)

def build_model(name: str, **kwargs):
    backbone_cls = factory[name]
    backbone = backbone_cls(**kwargs.pop("backbone", {}))
    seq_cfg = kwargs.get("seq_wrap", None)
    if seq_cfg and seq_cfg.get("enabled", False):
        backbone = SequenceWrapper(
            backbone,
            agg=seq_cfg.get("agg", "attn"),
            attn_dropout=seq_cfg.get("attn_dropout", 0.0),
        )

    return BasePVCModel(backbone=backbone, **{k:v for k,v in kwargs.items() if k != "backbone" and k != "seq_wrap"})
