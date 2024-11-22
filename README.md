# GRAPE: Guided-Reinforced Vision-Language-Action Preference Optimization
[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/openvla/openvla-7b)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
 
[**Installation**](#installation) | [**Training VLA model via TPO-LoRA**](#training-vla-model-via-tpo-lora) | [**Evaluating GRAPE**](#evaluating-grape) | [**Project Website**](https://openvla.github.io/)


<hr style="border: 2px solid gray;"></hr>

We release the codebase for the **GRAPE** framework, which includes the following components:

- **Customized Cost Generation**:  **GRAPE** decomposes complex manipulation tasks into multiple independent stages and leverages Vision-Language Models (VLMs) to generate relevant constraints for each stage.

- **Iterative Trajectory-wise Preference Optimization (TPO)**: Our iterative TPO framework enables refinement and improvement of VLA models over multiple training cycles.

- **Model Evaluation**:  Comprehensive evaluation of the **GRAPE** framework is supported on two benchmarks: **Simpler-Env** and **LIBERO**, providing rigorous testing for generalizability and performance.


Built on top of OpenVLA(https://github.com/openvla/openvla).


## Installation

Use the setup commands below to get started:

```bash
# Create and activate conda environment
conda create -n GRAPE python=3.10 -y
conda activate GRAPE
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  

# Clone and install the modified openvla repo
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

---

## Training VLA model via TPO-LoRA

In this section, we discuss training OpenVLA model via Trajectory-wise Preference Optimization(TPO). The main script for LoRA
training is `finetune.py`. 

Below we show an example of how you can train the main OpenVLA-SFT checkpoint ([`openvla-7b`](https://huggingface.co/openvla/openvla-7b))
via LoRA-TPO. Here we use a single A100 GPU with 80 GB VRAM. (Attention: We only support batchsize=1 and single-GPU training now, which means each batch has a pair of trajectories. We will support more settings and fully training in the future)

Now, launch the LoRA TPO script, as shown below. 

```bash
  torchrun --standalone --nnodes=1 --nproc-per-node 1 finetune.py \
  --vla_path "/data/zhaoyang_wang/projects/OpenVLA/openvla/ckpt/fullmodel" \
  --dataset_name "rlds_np_rollout" \
  --chosen_traj_dir <PATH TO CHOSEN TRAJECTORY DATASET> \
  --rejected_traj_dir <PATH TO REJECTED TRAJECTORY DATASET> \
  --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
  --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
  --lora_rank 32 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --image_aug False \
  --wandb_project <YOUR PROJECT NAME> \
  --wandb_entity <YOUR ENTITY> \
  --save_steps 1000
```

To LoRA fine-tune on a different dataset, you can download the dataset from our repo(URL TBD).

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
