# GRAPE: Guided-Reinforced Vision-Language-Action Preference Optimization
[![arXiv](https://img.shields.io/badge/arXiv-2406.09246-df2a2a.svg?style=for-the-badge)](https://arxiv.org/abs/2406.09246)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow?style=for-the-badge)](https://huggingface.co/openvla/openvla-7b)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/python-3.10-blue?style=for-the-badge)](https://www.python.org)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
 
[**Installation**](#installation) | [**Training VLA model via TPO-LoRA**](#training-vla-model-via-tpo-lora) | [**Datasets**](#datasets) | [**Evaluating GRAPE**](#evaluating-grape) | [**Project Website**](https://openvla.github.io/)


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

# Install the modified openvla repo
cd TPO-Train
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

Now, launch the TPO-LoRA script, as shown below. 

```bash
  torchrun --standalone --nnodes=1 --nproc-per-node 1 finetune.py \
  --vla_path <PATH TO REFERENCE MODEL> \
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

For details about chosen_traj and rejected_traj, You can refer  [**Datasets**](#datasets) . 

To TPO-LoRA on a sample dataset, you can download the dataset from our repo(URL TBD).



## Datasets

**To support RLDS format datasets, we designed our dataset in a special way. Specifically, when we built the chosen_traj and rejected_traj datasets, we ensured that their trajectories were paired one-to-one. That is, the nth trajectory data in chosen_traj corresponds to the nth trajectory data in rejected_traj, and they are from the same task with consistent initial state. We will release scripts to construct RLDS datasets for TPO in the future**

### Datasets Collection and Preference Generation

**Note that our Guided-cost Preference Generation is available in Simpler-Env. Support for the Real-World environment will be provided later.**


#### In Simpler-Env

Our data collection in Simpler-Env is embedded in Simpler-Env evaluation. You can collect trajectory data by modifying Simpler-Env files.

**What you need to do for data collection is:**
1. Overwrite ./Simpler-env/simpler_env/evaluation/maniskill2_evaluator.py with./Data Collection/maniskill2_evaluator.py.
2. Overwrite the modeling_prismatic.py in your tpo-model's folder with./Data Collection/modeling_prismatic.py.
3. Then you can refer to [**Simpler-Env**](#simpler-env) for running.


For preference generation, the relevant code could be found in /Data Collection/maniskill2_evaluator.py. The final GCPG reward of a trajectory will be shown in its filename. You can rank these trajectories in a task based on the GCPG reward. It is highly recommanded that you should modify **beta** and **threshold** for your experiments. The GCPG reward works well when $$R_{self}$$, $$R_{ext}$$ and $$I_{success}$$ contribute comparably to the final reward value.




#### In LIBERO

Our data collection in LIBERO is embedded in LIBERO evaluation, too. You can collect trajectory data by modifying LIBERO files.

**What you need to do is:**
1. Overwrite experiments/robot/libero/run_libero_eval.py with ./Data Collection/libero_data_collect.py.
2. Overwrite the modeling_prismatic.py in your tpo-model's folder with./Data Collection/modeling_prismatic.py.
3. Then you can refer to [**LIBERO**](#libero) for running.


## Evaluating GRAPE

We support two evaluation benchmarks in simulation environments: **Simpler-Env** and **LIBERO**


### Simpler-Env
#### Simpler-Env Setup

Note: We use Colab for our evaluation experiments. Settings in Colab and local GPU may be different. Please feel free to file a Github issue for any problems here.

Use the setup commands below to get started:

```bash

#  Install vulkan for rendering
apt-get install -yqq --no-install-recommends libvulkan-dev vulkan-tools
# below fixes some bugs introduced by some recent Colab changes
mkdir -p /usr/share/vulkan/icd.d
wget -q -P /usr/share/vulkan/icd.d https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/nvidia_icd.json
wget -q -O /usr/share/glvnd/egl_vendor.d/10_nvidia.json https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/10_nvidia.json

#  Install Real2Sim

pip install numpy==1.24.4
pip install orbax-checkpoint==0.4.4
pip install scipy==1.12.0
pip install keras==2.15.0
pip install tensorflow==2.15.1

# Install OpenVLA dependency

pip install torch==2.3.1 torchvision==0.18.1 timm==0.9.10 tokenizers==0.15.2 accelerate==0.32.1
pip install flash-attn==2.6.1 --no-build-isolation
pip install --quiet tf_agents
pip install --quiet mediapy
pip install peft


# Install Simpler-Env
cd Simpler-Env\ManiSkill2_real2sim
pip install -e .
cd ..
pip install -e .
```
#### Run Simpler-Env

```bash
python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/tpo_model" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 50 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/tpo_model" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 20 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/tpo_model" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutSpoonOnTableClothInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 --obj-variation-mode episode --obj-episode-range 0 20 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1

python simpler_env/main_inference.py --policy-model openvla --ckpt-path "/path/to/tpo_model" \
  --robot widowx_sink_camera_setup --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 100 \
  --env-name PutEggplantInBasketScene-v0 --scene-name bridge_table_1_v2 \
  --rgb-overlay-path ./ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png \
  --robot-init-x 0.127 0.127 1 --robot-init-y 0.06 0.06 1 --obj-variation-mode episode --obj-episode-range 0 20 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1


```

### LIBERO
#### LIBERO Setup
Clone and install the [LIBERO repo](https://github.com/Lifelong-Robot-Learning/LIBERO):

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

Additionally, install other required packages:
```bash
cd openvla
pip install -r experiments/robot/libero/libero_requirements.txt
```


#### Run LIBERO

To start evaluation , run one of the commands below. Each will automatically download the appropriate checkpoint listed above.

```bash
# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint <PATH TO YOUR TPO MODEL> \
  --task_suite_name libero_spatial \
  --center_crop True

# Launch LIBERO-Object evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint <PATH TO YOUR TPO MODEL> \
  --task_suite_name libero_object \
  --center_crop True

# Launch LIBERO-Goal evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint <PATH TO YOUR TPO MODEL> \
  --task_suite_name libero_goal \
  --center_crop True

# Launch LIBERO-10 (LIBERO-Long) evals
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint <PATH TO YOUR TPO MODEL> \
  --task_suite_name libero_10 \
  --center_crop True
```


#### Citation

If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/abs/):

```bibtex
@article{zhang24grape,
    title={OpenVLA: An Open-Source Vision-Language-Action Model},
    author={{Moo Jin} Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
    journal = {arXiv preprint arXiv:2406.09246},
    year={2024}
} 
```
