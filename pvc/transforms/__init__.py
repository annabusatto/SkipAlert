import os
import torch.nn as nn
from .gaussian_noise import GaussianNoise
from .amplitude_scale import AmplitudeScale
from .baseline_wander import BaselineWander
from .time_warp import TimeWarp
from .lead_dropout import LeadDropout
from .graph_blur import GraphLaplacianBlur

def default_augment(adj_path: str | None = None):
    if adj_path is None:
        # default to your .pt file
        adj_path = os.environ.get(
            "PVC_ADJ_PATH",
            os.path.join(os.getcwd(), "resources", "adjacency_matrix_tensor.pt"),
        )

    return nn.Sequential(
        GaussianNoise(20.0),
        AmplitudeScale(),
        BaselineWander(),
        GraphLaplacianBlur(
            adj_path=adj_path,
            mode="normalized",
            alpha=(0.03, 0.12),
            steps=1,
            p=0.5,
        ),
        TimeWarp(),
        LeadDropout(),
    )
