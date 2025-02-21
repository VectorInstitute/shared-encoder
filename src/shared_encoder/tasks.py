"""Contrastive pretraining task."""

import itertools
from functools import partial
from typing import Any, Optional, Union

import lightning as L  # noqa: N812
import numpy as np
import torch
import torch.distributed
import torch.distributed.nn
from mmlearn.conf import external_store
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality
from mmlearn.modules.losses import ContrastiveLoss
from mmlearn.tasks import ContrastivePretraining
from mmlearn.tasks.contrastive_pretraining import (
    AuxiliaryTaskSpec,
    EvaluationSpec,
    LossPairSpec,
    ModuleKeySpec,
    _unsupported_modality_error,
)
from mmlearn.tasks.hooks import EvaluationHooks
from torch import nn


@external_store(group="task", provider="shared_encoder")
class SharedEncoderContrastivePretraining(ContrastivePretraining):
    """Contrastive pretraining task with shared encoder.

    This class supports contrastive pretraining with `N` modalities of data. It
    allows the sharing of encoders, heads, and postprocessors across modalities.
    It also supports computing the contrastive loss between specified pairs of
    modalities, as well as training auxiliary tasks alongside the main contrastive
    pretraining task.

    Parameters
    ----------
    encoders : dict[str, nn.Module]
        A dictionary of encoders. The keys can be any string, including the names of
        any supported modalities. If the keys are not supported modalities, the
        `modality_module_mapping` parameter must be provided to map the encoders to
        specific modalities. The encoders are expected to take a dictionary of input
        values and return a list-like object with the first element being the encoded
        values. This first element is passed on to the heads or postprocessors and
        the remaining elements are ignored.
    heads : dict[str, Union[nn.Module, dict[str, nn.Module]]], optional, default=None
        A dictionary of modules that process the encoder outputs, usually projection
        heads. If the keys do not correspond to the name of a supported modality,
        the `modality_module_mapping` parameter must be provided. If any of the values
        are dictionaries, they will be wrapped in a `nn.Sequential` module. All
        head modules are expected to take a single input tensor and return a single
        output tensor.
    postprocessors : dict[str, Union[nn.Module, dict[str, nn.Module]]], optional, default=None
        A dictionary of modules that process the head outputs. If the keys do not
        correspond to the name of a supported modality, the `modality_module_mapping`
        parameter must be provided. If any of the values are dictionaries, they will
        be wrapped in a `nn.Sequential` module. All postprocessor modules are expected
        to take a single input tensor and return a single output tensor.
    modality_module_mapping : dict[str, ModuleKeySpec], optional, default=None
        A dictionary mapping modalities to encoders, heads, and postprocessors.
        Useful for reusing the same instance of a module across multiple modalities.
    optimizer : partial[torch.optim.Optimizer], optional, default=None
        The optimizer to use for training. This is expected to be a partial function,
        created using `functools.partial`, that takes the model parameters as the
        only required argument. If not provided, training will continue without an
        optimizer.
    lr_scheduler : Union[dict[str, Union[partial[torch.optim.lr_scheduler.LRScheduler], Any]], partial[torch.optim.lr_scheduler.LRScheduler]], optional, default=None
        The learning rate scheduler to use for training. This can be a partial function
        that takes the optimizer as the only required argument or a dictionary with
        a `scheduler` key that specifies the scheduler and an optional `extras` key
        that specifies additional arguments to pass to the scheduler. If not provided,
        the learning rate will not be adjusted during training.
    init_logit_scale : float, optional, default=1 / 0.07
        The initial value of the logit scale parameter. This is the log of the scale
        factor applied to the logits before computing the contrastive loss.
    max_logit_scale : float, optional, default=100
        The maximum value of the logit scale parameter. The logit scale parameter
        is clamped to the range [0, log(max_logit_scale)].
    learnable_logit_scale : bool, optional, default=True
        Whether the logit scale parameter is learnable. If set to False, the logit
        scale parameter is treated as a constant.
    loss : ContrastiveLoss, optional, default=None
        The loss function to use.
    modality_loss_pairs : list[LossPairSpec], optional, default=None
        A list of pairs of modalities to compute the contrastive loss between and
        the weight to apply to each pair.
    auxiliary_tasks : dict[str, AuxiliaryTaskSpec], optional, default=None
        Auxiliary tasks to run alongside the main contrastive pretraining task.
        The auxiliary task module is expected to be a partially-initialized instance
        of a `LightningModule` created using `functools.partial`, such that an
        initialized encoder can be passed as the only argument. The `modality`
        parameter specifies the modality of the encoder to use for the auxiliary task.
        The `loss_weight` parameter specifies the weight to apply to the auxiliary
        task loss.
    log_auxiliary_tasks_loss : bool, optional, default=False
        Whether to log the loss of auxiliary tasks to the main logger.
    compute_validation_loss : bool, optional, default=True
        Whether to compute the validation loss if a validation dataloader is provided.
        The loss function must be provided to compute the validation loss.
    compute_test_loss : bool, optional, default=True
        Whether to compute the test loss if a test dataloader is provided. The loss
        function must be provided to compute the test loss.
    evaluation_tasks : dict[str, EvaluationSpec], optional, default=None
        Evaluation tasks to run during validation, while training, and during testing.
    shared_layers : list[int], optional, default=None
        The list of layers to share across modalities. If not provided, no layers
        will be shared.

    """  # noqa: W505

    def __init__(  # noqa: PLR0912, PLR0915
        self,
        encoders: dict[str, nn.Module],
        heads: Optional[dict[str, Union[nn.Module, dict[str, nn.Module]]]] = None,
        postprocessors: Optional[
            dict[str, Union[nn.Module, dict[str, nn.Module]]]
        ] = None,
        modality_module_mapping: Optional[dict[str, ModuleKeySpec]] = None,
        optimizer: Optional[partial[torch.optim.Optimizer]] = None,
        lr_scheduler: Optional[
            Union[
                dict[str, Union[partial[torch.optim.lr_scheduler.LRScheduler], Any]],
                partial[torch.optim.lr_scheduler.LRScheduler],
            ]
        ] = None,
        init_logit_scale: float = 1 / 0.07,
        max_logit_scale: float = 100,
        learnable_logit_scale: bool = True,
        loss: Optional[ContrastiveLoss] = None,
        modality_loss_pairs: Optional[list[LossPairSpec]] = None,
        auxiliary_tasks: Optional[dict[str, AuxiliaryTaskSpec]] = None,
        log_auxiliary_tasks_loss: bool = False,
        compute_validation_loss: bool = True,
        compute_test_loss: bool = True,
        evaluation_tasks: Optional[dict[str, EvaluationSpec]] = None,
        shared_layers: Optional[list[int]] = None,
    ) -> None:
        """Initialize the module."""
        L.LightningModule.__init__(self)

        if shared_layers is None:
            shared_layers = []

        modality_keys = list(encoders.keys())
        if len(modality_keys) > 1:
            for i in range(1, len(modality_keys)):
                source_key = modality_keys[0]
                target_key = modality_keys[i]
                for layer in shared_layers:
                    encoders[target_key].transformer.resblocks[layer] = encoders[
                        source_key
                    ].transformer.resblocks[layer]

        if modality_module_mapping is None:
            # assume all the module dictionaries use the same keys corresponding
            # to modalities
            modality_module_mapping = {}
            for key in encoders:
                modality_module_mapping[key] = ModuleKeySpec(
                    encoder_key=key,
                    head_key=key,
                    postprocessor_key=key,
                )

        # match modalities to encoders, heads, and postprocessors
        modality_encoder_mapping: dict[str, Optional[str]] = {}
        modality_head_mapping: dict[str, Optional[str]] = {}
        modality_postprocessor_mapping: dict[str, Optional[str]] = {}
        for modality_key, module_mapping in modality_module_mapping.items():
            if not Modalities.has_modality(modality_key):
                raise ValueError(_unsupported_modality_error.format(modality_key))
            modality_encoder_mapping[modality_key] = module_mapping.encoder_key
            modality_head_mapping[modality_key] = module_mapping.head_key
            modality_postprocessor_mapping[modality_key] = (
                module_mapping.postprocessor_key
            )

        # ensure all modules are mapped to a modality
        for key in encoders:
            if key not in modality_encoder_mapping.values():
                if not Modalities.has_modality(key):
                    raise ValueError(_unsupported_modality_error.format(key))
                modality_encoder_mapping[key] = key

        if heads is not None:
            for key in heads:
                if key not in modality_head_mapping.values():
                    if not Modalities.has_modality(key):
                        raise ValueError(_unsupported_modality_error.format(key))
                    modality_head_mapping[key] = key

        if postprocessors is not None:
            for key in postprocessors:
                if key not in modality_postprocessor_mapping.values():
                    if not Modalities.has_modality(key):
                        raise ValueError(_unsupported_modality_error.format(key))
                    modality_postprocessor_mapping[key] = key

        self._available_modalities: list[Modality] = [
            Modalities.get_modality(modality_key)
            for modality_key in modality_encoder_mapping
        ]
        assert len(self._available_modalities) >= 2, (
            "Expected at least two modalities to be available. "
        )

        self.encoders = nn.ModuleDict(
            {
                Modalities.get_modality(modality_key).name: encoders[encoder_key]
                for modality_key, encoder_key in modality_encoder_mapping.items()
                if encoder_key is not None
            }
        )
        self.heads = None
        if heads is not None:
            self.heads = nn.ModuleDict(
                {
                    Modalities.get_modality(modality_key).name: heads[head_key]
                    if isinstance(heads[head_key], nn.Module)
                    else nn.Sequential(*heads[head_key].values())
                    for modality_key, head_key in modality_head_mapping.items()
                    if head_key is not None
                }
            )

        self.postprocessors = None
        if postprocessors is not None:
            self.postprocessors = nn.ModuleDict(
                {
                    Modalities.get_modality(modality_key).name: postprocessors[
                        postprocessor_key
                    ]
                    if isinstance(postprocessors[postprocessor_key], nn.Module)
                    else nn.Sequential(*postprocessors[postprocessor_key].values())
                    for modality_key, postprocessor_key in modality_postprocessor_mapping.items()
                    if postprocessor_key is not None
                }
            )

        # set up logit scaling
        log_logit_scale = torch.ones([]) * np.log(init_logit_scale)
        self.max_logit_scale = max_logit_scale
        self.learnable_logit_scale = learnable_logit_scale

        if self.learnable_logit_scale:
            self.log_logit_scale = torch.nn.Parameter(
                log_logit_scale, requires_grad=True
            )
        else:
            self.register_buffer("log_logit_scale", log_logit_scale)

        if modality_loss_pairs is None:
            modality_loss_pairs = [
                LossPairSpec(modalities=(m1.name, m2.name))
                for m1, m2 in itertools.combinations(self._available_modalities, 2)
            ]

        for modality_pair in modality_loss_pairs:
            if not all(
                Modalities.get_modality(modality) in self._available_modalities
                for modality in modality_pair.modalities
            ):
                raise ValueError(
                    "Found unspecified modality in the loss pair specification "
                    f"{modality_pair.modalities}. Available modalities are "
                    f"{self._available_modalities}."
                )
        self.modality_loss_pairs = modality_loss_pairs

        self.aux_task_specs = auxiliary_tasks or {}
        self.auxiliary_tasks: dict[str, L.LightningModule] = {}
        for task_name, task_spec in self.aux_task_specs.items():
            if not Modalities.has_modality(task_spec.modality):
                raise ValueError(
                    f"Found unsupported modality `{task_spec.modality}` in the auxiliary tasks. "
                    f"Available modalities are {self._available_modalities}."
                )
            if not isinstance(task_spec.task, partial):
                raise TypeError(
                    f"Expected auxiliary task to be a partial function, but got {type(task_spec.task)}."
                )

            self.auxiliary_tasks[task_name] = task_spec.task(
                self.encoders[Modalities.get_modality(task_spec.modality).name]
            )

        if loss is None and (compute_validation_loss or compute_test_loss):
            raise ValueError(
                "Loss function must be provided to compute validation or test loss."
            )

        self.loss_fn = loss
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.log_auxiliary_tasks_loss = log_auxiliary_tasks_loss
        self.compute_validation_loss = compute_validation_loss
        self.compute_test_loss = compute_test_loss

        if evaluation_tasks is not None:
            for eval_task_spec in evaluation_tasks.values():
                if not isinstance(eval_task_spec.task, EvaluationHooks):
                    raise TypeError(
                        f"Expected {eval_task_spec.task} to be an instance of `EvaluationHooks` "
                        f"but got {type(eval_task_spec.task)}."
                    )
        self.evaluation_tasks = evaluation_tasks

    def encode(
        self, inputs: dict[str, Any], modality: Modality, normalize: bool = False
    ) -> torch.Tensor:
        """Encode the input values for the given modality.

        Parameters
        ----------
        inputs : dict[str, Any]
            Input values.
        modality : Modality
            The modality to encode.
        normalize : bool, optional, default=False
            Whether to apply L2 normalization to the output (after the head and
            postprocessor layers, if present).

        Returns
        -------
        torch.Tensor
            The encoded values for the specified modality.
        """
        output = self.encoders[modality.name](inputs[modality.name])[0]

        if self.heads and modality.name in self.heads:
            output = self.heads[modality.name](output)

        if self.postprocessors and modality.name in self.postprocessors:
            output = self.postprocessors[modality.name](output)

        if normalize:
            output = torch.nn.functional.normalize(output, p=2, dim=-1)

        return output
