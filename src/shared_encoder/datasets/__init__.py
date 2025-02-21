"""Datasets for training and evaluation."""

from typing import Literal

import torch
import torchvision.transforms.v2 as transforms
from mmlearn.conf import external_store
from timm.data.transforms import ResizeKeepRatio

from .deepeyenet import DeepEyeNet
from .mimiciv_cxr import MIMICIVCXR
from .pmcoa import PMC2M, PMCOA
from .quilt import Quilt


@external_store(group="datasets/transforms")
def med_clip_vision_transform(
    image_crop_size: int = 224, job_type: Literal["train", "eval"] = "train"
) -> transforms.Compose:
    """Return transforms for training/evaluating CLIP with medical images.

    Parameters
    ----------
    image_crop_size : int, default=224
        Size of the image crop.
    job_type : {"train", "eval"}, default="train"
        Type of the job (training or evaluation) for which the transforms are needed.

    Returns
    -------
    transforms.Compose
        Composed transforms for training CLIP with medical images.
    """
    if job_type == "train":
        return transforms.Compose(
            [
                ResizeKeepRatio(512, interpolation="bicubic"),
                transforms.RandomCrop(image_crop_size),
                transforms.RGB(),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
    return transforms.Compose(
        [
            ResizeKeepRatio(image_crop_size, interpolation="bicubic"),
            transforms.CenterCrop(image_crop_size),
            transforms.RGB(),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


__all__ = [
    "DeepEyeNet",
    "MIMICIVCXR",
    "PMC2M",
    "PMCOA",
    "Quilt",
    "med_clip_vision_transform",
]
