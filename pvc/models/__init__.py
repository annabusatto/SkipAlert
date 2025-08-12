from importlib import import_module

# list all backbone files once
_names = ["resnet", "tcn", "inception", "tst", "wave_cnn", "lstm_attn", "conv_attn"]

factory = {}
for n in _names:
    mod = import_module(f".backbones.{n}", package=__name__)
    cls_name = [c for c in mod.__dict__ if c.endswith("Backbone")][0]
    factory[n] = getattr(mod, cls_name)
