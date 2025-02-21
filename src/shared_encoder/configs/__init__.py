"""Config entrypoint for experiments - training and evaluation."""

from shared_encoder.datasets import *
from shared_encoder.encoders import MMCLIPText, MMCLIPVision
from shared_encoder.tasks import SharedEncoderContrastivePretraining
