# GRAPE: Guided-Reinforced Vision-Language-Action Preference Optimization
[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/openvla/openvla-7b)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
 
[**Getting Started**](#getting-started) | [**Installation**](#installation) | [**Fine-Tuning OpenVLA via LoRA**](#fine-tuning-openvla-via-lora) | [**Fully Fine-Tuning OpenVLA**](#fully-fine-tuning-openvla) |
[**Training VLAs from Scratch**](#training-vlas-from-scratch) | [**Evaluating OpenVLA**](#evaluating-openvla) | [**Project Website**](https://openvla.github.io/)


<hr style="border: 2px solid gray;"></hr>


A simple and scalable codebase for Trajectory-wise DPO() vision-language-action models (VLAs) for generalist robotic 

We introduce **T**rajectory-wise **P**reference **O**ptimization (TPO) to align VLA policies on a trajectory level by implicitly modeling reward from both successful and failure trials, boosting generalizability to diverse tasks. 
manipulation:

- **Different Dataset Mixtures**: We natively support arbitrary datasets in RLDS format, including arbitrary mixtures of
  data from the [Open X-Embodiment Dataset](https://robotics-transformer-x.github.io/).
- **Easy Scaling**: Powered by PyTorch FSDP and Flash-Attention, we can quickly and efficiently train models from 1B - 
  34B parameters, with easily adaptable model architectures.
- **Native Fine-Tuning Support**: Built-in support (with examples) for various forms of fine-tuning (full, 
  partial, LoRA).

Built on top of [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms).

## Getting Started

To get started with loading and running OpenVLA models for inference, we provide a lightweight interface that leverages
HuggingFace `transformers` AutoClasses, with minimal dependencies.

For example, to load `openvla-7b` for zero-shot instruction following in the
[BridgeData V2 environments](https://rail-berkeley.github.io/bridgedata/) with a WidowX robot:

```python
# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b", 
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

# Grab image input & format prompt
image: Image.Image = get_from_camera(...)
prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

# Execute...
robot.act(action, ...)
```

We also provide an [example script for fine-tuning OpenVLA models for new tasks and 
embodiments](./vla-scripts/finetune.py); this script supports different fine-tuning modes -- including (quantized) 
low-rank adaptation (LoRA) supported by [HuggingFace's PEFT library](https://huggingface.co/docs/peft/en/index). 




---

## Installation

> **Note**: These installation instructions are for full-scale pretraining (and distributed fine-tuning); if looking to
  just run inference with OpenVLA models (or perform lightweight fine-tuning), see instructions above!

This repository was built using Python 3.10, but should be backwards compatible with any Python >= 3.8. We require
PyTorch 2.2.* -- installation instructions [can be found here](https://pytorch.org/get-started/locally/). The latest 
version of this repository was developed and thoroughly tested with:
  - PyTorch 2.2.0, torchvision 0.17.0, transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, and flash-attn 2.5.5

**[5/21/24] Note**: Following reported regressions and breaking changes in later versions of `transformers`, `timm`, and
`tokenizers` we explicitly pin the above versions of the dependencies. We are working on implementing thorough tests, 
and plan on relaxing these constraints as soon as we can.

Use the setup commands below to get started:

```bash
# Create and activate conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch. Below is a sample command to do this, but you should check the following link
# to find installation instructions that are specific to your compute platform:
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  # UPDATE ME!

# Clone and install the openvla repo
git clone https://github.com/zzj1111/Openvla-GRAPE.git
cd Openvla-GRAPE
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation
```

If you run into any problems during the installation process, please file a GitHub Issue.

**Note:** See `vla-scripts/` for full training and verification scripts for OpenVLA models. Note that `scripts/` is
mostly a holdover from the original (base) `prismatic-vlms` repository, with support for training and evaluating
visually-conditioned language models; while you can use this repo to train VLMs AND VLAs, note that trying to generate
language (via `scripts/generate.py`) with existing OpenVLA models will not work (as we only train current OpenVLA models
to generate actions, and actions alone).

## Fine-Tuning OpenVLA via LoRA

In this section, we discuss fine-tuning OpenVLA using Low-Rank Adaptation (LoRA) via the Hugging Face `transformers` library,
which is recommended if you do not have sufficient compute to fully fine-tune a 7B-parameter model. The main script for LoRA
fine-tuning is `vla-scripts/finetune.py`. (If you instead wish to do full fine-tuning, please see the
[Fully Fine-Tuning OpenVLA](#fully-fine-tuning-openvla) section.)

Below we show an example of how you can fine-tune the main OpenVLA checkpoint ([`openvla-7b`](https://huggingface.co/openvla/openvla-7b))
via LoRA. Here we fine-tune on [BridgeData V2](https://rail-berkeley.github.io/bridgedata/) using a single A100
GPU with 80 GB VRAM. (You can also fine-tune with a smaller GPU, as long as it has at least ~27 GB of memory,
by modifying the batch size.)

First, download the BridgeData V2 dataset:

```bash
# Change directory to your base datasets folder
cd <PATH TO BASE DATASETS DIR>

# Download the full dataset (124 GB)
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/

# Rename the dataset to `bridge_orig` (NOTE: Omitting this step may lead to runtime errors later)
mv bridge_dataset bridge_orig
```

Now, launch the LoRA fine-tuning script, as shown below. Note that `--batch_size==16` with `--grad_accumulation_steps==1`
requires ~72 GB GPU memory. If you have a smaller GPU, you should reduce `--batch_size` and increase `--grad_accumulation_steps`
to maintain an effective batch size that is large enough for stable training. If you have multiple GPUs and wish to train via
PyTorch Distributed Data Parallel (DDP), simply set `--nproc-per-node` in the `torchrun` command below to the number of available GPUs.

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <PATH TO BASE DATASETS DIR> \
  --dataset_name bridge_orig \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug <True or False> \
  --wandb_project <PROJECT> \
  --wandb_entity <ENTITY> \
  --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>
```

Note: If you set `--image_aug==False` in the command above, you will observe nearly 100% `action_accuracy` in the training logs,
since the [`openvla-7b`](https://huggingface.co/openvla/openvla-7b) model is already pretrained (without augmentations) on a
superset of datasets that includes BridgeData V2.

To LoRA fine-tune on a different dataset, you can download the dataset from the [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/)
mixture (see [this custom script](https://github.com/moojink/rlds_dataset_mod/blob/main/prepare_open_x.sh) for an example of how to download datasets
from OXE). Alternatively, if you have a custom dataset that is not part of OXE, you can either (a) convert the dataset to the RLDS format which is
compatible with our fine-tuning script (see [this repo](https://github.com/kpertsch/rlds_dataset_builder) for instructions on this), or (b) use your own
custom PyTorch Dataset wrapper (see comments in `vla-scripts/finetune.py` for instructions). We recommend option (a) for most users; the RLDS dataset and
dataloader are tested more extensively since we used these for all of our pretraining and fine-tuning experiments.

For option (a), after you converted your dataset to RLDS, you need to register it with our data loader, by registering a dataset
config [here](prismatic/vla/datasets/rlds/oxe/configs.py#L54) and a dataset transform function [here](prismatic/vla/datasets/rlds/oxe/transforms.py#L828).

Once you have integrated your new dataset, you can launch LoRA fine-tuning with the same `vla-scripts/finetune.py` script above. If you run into any issues,
please visit the [VLA Troubleshooting](#vla-troubleshooting) section or search for a similar issue in the [OpenVLA GitHub Issues page](https://github.com/openvla/openvla/issues?q=)
(including "Closed" issues). If you cannot find a similar issue there, feel free to create a new issue.


## Repository Structure

High-level overview of repository/project file-tree:

+ `prismatic` - Package source; provides core utilities for model loading, training, data preprocessing, etc.
+ `vla-scripts/` - Core scripts for training, fine-tuning, and deploying VLAs.
+ `experiments/` - Code for evaluating OpenVLA policies in robot environments.
+ `LICENSE` - All code is made available under the MIT License; happy hacking!
+ `Makefile` - Top-level Makefile (by default, supports linting - checking & auto-fix); extend as needed.
+ `pyproject.toml` - Full project configuration details (including dependencies), as well as tool configurations.
+ `README.md` - You are here!

---


#### Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/2406.09246):

```bibtex
@article{kim24openvla,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
} 
```
