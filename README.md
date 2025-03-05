# A Shared Encoder Approach to Multimodal Representation Learning

This repository contains code to reproduce the paper [A Shared Encoder Approach to Multimodal Representation Learning](https://arxiv.org/abs/2503.01654).

## Setup

To reproduce experiments, first set up a local copy of the repository by cloning it:

```bash
git clone https://github.com/VectorInstitute/shared_encoder.git
```

Then install [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) and run the following command to install
the dependencies for the project:

```bash
uv sync -n --dev
```

## Running Experiments

### Pretraining

Prior to running a pretraining job, please set the following environment variables:

```bash
export PMCOA_ROOT_DIR=/path/to/PMC-OA
```

Then run the following command to pretrain the model:

```bash
mmlearn_run --multirun \
    hydra.launcher.mem_per_cpu=5G \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.partition=a40 \
    hydra.launcher.qos=normal \
    hydra.launcher.gres=gpu:4 \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.nodes=2 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=960 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://shared_encoder.configs]' \
    +experiment=PMC-OA-SHARE-MT-20 \
    trainer.num_nodes=2 \
    experiment_name=PMC-OA-SHARE-MT-20-12Layers
```

**Note**: This command is will schedule a job on a SLURM cluster. If you are running the job locally, please remove
all arguments that start with `hydra.launcher` as well as the `--multirun` flag.

### Evaluation

Run the following command to evaluate a pretrained model on the test set of the PMC-OA dataset:

```bash
mmlearn_run --multirun hydra.launcher.mem_per_cpu=5G \
    hydra.launcher.cpus_per_task=10 \
    hydra.launcher.partition=rtx6000 \
    hydra.launcher.qos=normal \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=60 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    hydra.searchpath=[pkg://shared_encoder.configs] \
    +experiment=PMC-OA-SHARE-MT-20 \
    job_type=eval \
    +datasets@datasets.test=PMCOA \
    datasets.test.split=test \
    +datasets/transforms@datasets.test.transform=med_clip_vision_transform \
    +datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=HFCLIPTokenizer \
    experiment_name=ZSR-PMC-OA-SHARE-MT-20-12Layers \
    resume_from_checkpoint=<path_to_checkpoint>
```

**Note**: To evaluate the model on MIMIC-CXR, DeepEyeNet and/or Quilt, please set one or more of the following environment variables:

```bash
export DEY_ROOT_DIR=/path/to/DeepEyeNet/dataset
export MIMICIVCXR_ROOT_DIR=/path/to/MIMIC-CXR/dataset
export QUILT_ROOT_DIR=/path/to/Quilt
```
