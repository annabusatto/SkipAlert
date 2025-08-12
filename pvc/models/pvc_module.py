# pvc/models/pvc_module.py
from .common import BasePVCModel
from . import factory                     # registry (see below)

def build_model(name: str, **kwargs):
    backbone_cls = factory[name]
    backbone = backbone_cls(**kwargs.pop("backbone", {}))
    return BasePVCModel(backbone=backbone, **kwargs)
