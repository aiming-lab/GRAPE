"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import cv2
import imageio
import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

def write_video(video_path, images, fps=30, log_file=None):
    # create dir
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    # save video
    video_writer = imageio.get_writer(video_path, fps=fps)
    for img in images:
 
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        video_writer.append_data(img)
    video_writer.close()
    
    print(f"video saved to {video_path}")
    if log_file is not None:
        log_file.write(f"video saved to {video_path}\n")

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    print("norm:",model.norm_stats)
    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # get default LIBERO initial state
        initial_states = task_suite.get_task_init_states(task_id)

        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)


        task_episodes, task_successes = 0, 0
        total_episodes = 0

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            print(f"\ntask: {task_description}")
            log_file.write(f"\ntask: {task_description}\n")

        
            task_dir = os.path.join("./results", task_description)
            os.makedirs(task_dir, exist_ok=True)

         
            episode_dir = os.path.join(task_dir, f"episode_{episode_idx}")
            os.makedirs(episode_dir, exist_ok=True)

            initial_state = initial_states[episode_idx]

            episode_results = {}
            
            traj_num=0
            # collect a success traj and a failure traj in each episode
            while ('success' not in episode_results or 'failure' not in episode_results or traj_num<=5) and (traj_num<=10)  :
                
                traj_num+=1
               
                env.reset()
                obs = env.set_init_state(initial_state)

                t = 0
                total_data = []
                replay_images = []
                success = 'failure'  

                print(f"start new trial,episode {episode_idx}...")
                log_file.write(f"start new trial,episode {episode_idx}...\n")

               
                if cfg.task_suite_name == "libero_spatial":
                    max_steps = 220  # max step length 193
                elif cfg.task_suite_name == "libero_object":
                    max_steps = 280  # max step length 254
                elif cfg.task_suite_name == "libero_goal":
                    max_steps = 300  # max step length 270
                elif cfg.task_suite_name == "libero_10":
                    max_steps = 520  # max step length 505
                elif cfg.task_suite_name == "libero_90":
                    max_steps = 400  # max step length 373

                # start to sample
                while t < max_steps + cfg.num_steps_wait:
                    try:
                       
                        if t < cfg.num_steps_wait:
                            obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                            t += 1
                            continue

                        
                        img = get_libero_image(obs, resize_size)

                        
                        replay_images.append(img)

                       
                        observation = {
                            "full_image": img,
                            "state": np.concatenate(
                                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                            ),
                        }

                        
                        action = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            processor=processor,
                        )

                        
                        action = normalize_gripper_action(action, binarize=True)

                       
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)

                       
                        obs, reward, done, info = env.step(action.tolist())

                       
                        data_point = {
                            'images': img,
                            'prompt': task_description,
                            'action': action.tolist()
                        }
                        total_data.append(data_point)

                        print(t, info)
                        print("action:", action)

                        if done:
                            
                            success = 'success' 
                            print(f"{success} in trial,episode {episode_idx}")
                            log_file.write(f"{success} in trial,episode {episode_idx}\n")
                            break 

                        t += 1

                    except Exception as e:
                        print(f"error: {e}")
                        log_file.write(f"error: {e}\n")
                        break


                if success not in episode_results:
                    data_filename = f"{success}_episode_{episode_idx}.npy"
                    video_filename = f"{success}_episode_{episode_idx}.mp4"
                    npy_path = os.path.join(episode_dir, data_filename)
                    np.save(npy_path, total_data, allow_pickle=True)

                    video_path = os.path.join(episode_dir, video_filename)
                    write_video(video_path, replay_images, fps=20)

                    episode_results[success] = {
                        'data': total_data,
                        'images': replay_images,
                        'npy_path': npy_path
                    }

                    print(f"{success} saved in episode {episode_idx}")
                    log_file.write(f"{success} saved in episode {episode_idx}\n")
                else:
                    print(f"{success} exists, episode {episode_idx}, new trial...")
                    log_file.write(f"{success} exists, episode {episode_idx}, new trial...\n")


            task_episodes += 1  # trajectories collected in one episode
            total_episodes += 1

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)


if __name__ == "__main__":
    eval_libero()
