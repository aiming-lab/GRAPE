"""
Evaluate a model on ManiSkill2 environment.
"""
import math
import os
import json
import numpy as np
from PIL import Image
from transforms3d.euler import quat2euler
import torch.nn.functional as F
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video

def stage1_subgoal_constraint1(end_effector, keypoints):  #
    """Align end-effector with the carrot's center."""
    carrot_center = keypoints[0]  # Assuming keypoint[0] is carrot center
    path_cost = np.linalg.norm(end_effector - carrot_center)
    return path_cost

def stage1_collision_constraint1(end_effector, keypoints):
    """Ensure the end-effector approaches from above."""
    carrot_center = keypoints[0]
    # Check if the z-coordinate of the end-effector is higher than the carrot's z-coordinate
    collision_cost = 0 if end_effector[2] > carrot_center[2] else 1  # Penalize if below the carrot
    return collision_cost

def stage1_grasp_constraint(grasp_status, is_src_obj_grasped):
    """Grasp the carrot."""
    grasp_cost = 0 if is_src_obj_grasped else 1  # Grasp cost is incurred when the carrot is grasped
    return grasp_cost


### Stage 2: Move carrot to plate
# The carrot must stay grasped and avoid collisions.
def stage2_grasp_constraint(grasp_status, is_src_obj_grasped, src_on_target):
    """Ensure the carrot remains grasped during the move."""
    grasp_cost = 0 if is_src_obj_grasped else 1  # Carrot must remain grasped
    return grasp_cost

def stage2_collision_constraint(end_effector, keypoints):
    """Ensure the carrot is aligned above the plate."""
    carrot_center = keypoints[0]
    plate_center = keypoints[1]  # Assuming keypoint[1] is the plate center
    collision_cost = np.linalg.norm(carrot_center[:2] - plate_center[:2])  # Only consider x and y axes
    return collision_cost


### Stage 3: Drop carrot on plate
# Ensure the carrot is placed on the plate and avoid collision.
def stage3_path_constraint(end_effector, keypoints):
    """Place the carrot on the plate."""
    carrot_center = keypoints[0]
    plate_center = keypoints[1]
    path_cost = np.linalg.norm(carrot_center - plate_center)  # Ensure carrot is on the plate center
    return path_cost

def stage3_collision_constraint(end_effector, keypoints):
    """Ensure end-effector moves away after placing the carrot."""
    carrot_center = keypoints[0]
    # Check if the end-effector moves above and away after placing
    collision_cost = 0 if end_effector[2] > carrot_center[2] else 1
    return collision_cost
    

def cal_cost(end_effector, keypoints, stage, info):

    cost={}
    grasp_status=info['is_src_obj_grasped']
    is_src_obj_grasped=info['is_src_obj_grasped']
    src_on_target=info['src_on_target']

    if(stage==1):
      cost['path_cost']=stage1_subgoal_constraint1(end_effector, keypoints)
      cost['col_cost']=stage1_collision_constraint1(end_effector, keypoints)
      cost['grasp_cost']=stage1_grasp_constraint(grasp_status, is_src_obj_grasped)


    elif(stage==2):
      cost['grasp_cost']=stage2_grasp_constraint(grasp_status, is_src_obj_grasped, src_on_target)
      cost['col_cost']=stage2_collision_constraint(end_effector, keypoints)


    elif(stage==3):
      cost['path_cost']=stage3_path_constraint(end_effector, keypoints)
      cost['col_cost']=stage3_collision_constraint(end_effector, keypoints)
    print(cost)
    cost_sum=0
    for k,v in cost.items():
      cost_sum+=v
    return cost_sum,cost,stage

def external_score_option1(cost_matrix,alpha,threshold,stage_num):

    matrix = np.array([
        cost_matrix['col_cost'],
        cost_matrix['grasp_cost'],
        cost_matrix['path_cost']
    ])

    transposed_matrix = matrix.T
    cost_matrix=transposed_matrix
    
    ex_score=1

    for stage in range (stage_num):
        stage_score=1
        panel=0

        for cons in range(3):
            panel+=alpha[cons]*max(0,cost_matrix[stage][cons]-threshold[stage][cons])
        stage_score-=panel

        ex_score*=stage_score

    return ex_score


def external_score_option2(cost_matrix,beta,threshold,stage_num):
    matrix = np.array([
        cost_matrix['col_cost'],
        cost_matrix['grasp_cost'],
        cost_matrix['path_cost']
    ])
    

    transposed_matrix = matrix.T
    cost_matrix=transposed_matrix
    print(cost_matrix)
    ex_score=1
    stage_score=[]

    for stage in range (stage_num):
        stage_score=0
        panel=0
        
        for cons in range(3):
            panel-=beta[cons]*max(0,cost_matrix[stage][cons]-threshold[stage][cons])
        stage_score=np.exp(panel)

        ex_score*=stage_score

    return ex_score    







def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):

    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }

    episode_results = {}
    num_success=0
    num_fail=0
    failure_attempts=0

    while (('success' not in episode_results or 'failure' not in episode_results) or (num_fail+num_success)<=5) and (num_fail+num_success)<=10:

        obs, reset_info = env.reset(options=env_reset_options)
        # for long-horizon environments, we check if the current subtask is the final subtask
        is_final_subtask = env.unwrapped.is_final_subtask() 

        # Obtain language instruction
        if instruction is not None:
            task_description = instruction
        else:
            # get default language instruction
            task_description = env.unwrapped.get_language_instruction()
        print(task_description)

        task_dir = os.path.join("./results", task_description)
        os.makedirs(task_dir, exist_ok=True)

        episode_dir = os.path.join(task_dir, f"episode_{obj_episode_id}")  
        os.makedirs(episode_dir, exist_ok=True)

        # Initialize logging
        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images = [image]
        predicted_actions = []
        predicted_terminated, done, truncated = False, False, False

        # Initialize model
        model.reset(task_description)

        timestep = 0
        success = "failure"
        stage=1
        cost=0

        #diffenrent cost in stage 1,2,3
        cost_sum_dict={
        'col_cost': [0,0,0],
        'grasp_cost': [0,0,0],
        'path_cost': [0,0,0],
            }
        alpha=[1,1,1]
        beta=[0.01,0.01,0.01]
        threshold=[ [20,40,2],
                    [0,5,2],
                    [0,10,2] ]      # Note that you should modify these parameters for your task. These parameters vary significantly in different tasks.
        success_score=0
        inner_score=1        
        done_counter = 0
        total_data=[]


        # Step the environment
        while not (predicted_terminated or truncated):
            # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
            raw_action, action = model.step(image, task_description)
            inner_score*=max(model.vla.p,0.05)
            action_vector = np.concatenate([
                raw_action["world_vector"], 
                raw_action["rotation_delta"], 
                raw_action["open_gripper"]
            ])
            predicted_actions.append(action_vector)

            predicted_terminated = bool(action["terminate_episode"][0] > 0)
            if predicted_terminated:
                if not is_final_subtask:
                    # advance the environment to the next subtask
                    predicted_terminated = False
                    env.advance_to_next_subtask()


            # step the environment
            obs, reward, done, truncated, info = env.step(
                np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
            )
            obs_pose=obs['agent']['controller']['arm']['target_pose'][:3]
            source_pose=reset_info['episode_source_obj_init_pose_wrt_robot_base']
            tar_pose=reset_info['episode_target_obj_init_pose_wrt_robot_base'] 
            if(info["moved_correct_obj"]==False):
                stage=1    

            elif(info["moved_correct_obj"]==True and info['is_src_obj_grasped']==True):
                stage=2

            elif(info['is_src_obj_grasped']==True and info['consecutive_grasp']==True and info["src_on_target"]== False):
                stage=3
            keypoints=[np.array(source_pose.p),np.array(tar_pose.p)]
           # print(obs_pose,keypoints)
            cost_step,cost_dict,stage=cal_cost(end_effector=obs_pose,keypoints=keypoints,stage=stage,info=info)
            cost+=cost_step

            for k,v in cost_dict.items():
                cost_sum_dict[k][stage-1]+=v



            if done:

                done_counter += 1
                if done_counter >= 6:
                    success = 'success'
                    break 
            else:
                done_counter = 0
                
            # success = "success" if done else "failure"

            new_task_description = env.unwrapped.get_language_instruction()
            if new_task_description != task_description:
                task_description = new_task_description
                print(task_description)
            is_final_subtask = env.unwrapped.is_final_subtask()

            print(timestep, info)
            
            image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
            images.append(image)
            timestep += 1

            img = Image.fromarray(image)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            if success == 'success':
                subfolder = 'success'
            else:
                subfolder = 'failure'




            frame_path = os.path.join(episode_dir, f"step_{timestep}.png")


            data_point = {
            'images': img,
            'prompt': task_description,
            'action': action_vector.tolist()
            }
            total_data.append(data_point)


        success_score=1 if success=='success' else 0
        external_score_1=external_score_option1(cost_sum_dict,alpha=alpha,threshold=threshold,stage_num=3)
        external_score_2=external_score_option2(cost_sum_dict,beta=beta,threshold=threshold,stage_num=3)
        if(inner_score==0):
          final_traj_score_2=(external_score_2)+5*success_score-1
        else:
          final_traj_score_2=0.01*math.log(inner_score)+(external_score_2)+5*success_score
        episode_stats = info.get("episode_stats", {})
       # print("f_2:",final_traj_score_2)
        print("R_self:",inner_score,"R_ext:",external_score_2,"I_success:",success_score)
        if success == 'failure':

            failure_attempts += 1  
            if failure_attempts >= 50:
                break 
        print("failure:",failure_attempts)
        cost_int=int(final_traj_score_2)
        if(success=='success'):
            num_success+=1
            episode_results[success] = {
            'data': total_data,
            'images': images,
            'npy_path': os.path.join(episode_dir, f"{success}_episode_{obj_episode_id}_success_num_{num_success}_cost_{cost_int}.npy")
        }
        elif success=='failure':
            num_fail+=1
            episode_results[success] = {
            'data': total_data,
            'images': images,
            'npy_path': os.path.join(episode_dir, f"{success}_episode_{obj_episode_id}_fail_num_{num_fail}_cost_{cost_int}.npy")
            }
        
        np.save(episode_results[success]['npy_path'], total_data, allow_pickle=True)




        # npy_path=os.path.join(episode_dir, f"episode_{obj_episode_id}.npy")
        # np.save(npy_path,total_data, allow_pickle=True)

        episode_stats = info.get("episode_stats", {})


        
        video_name = f"{success}_episode_{obj_episode_id}"
        # for k,v in cost_sum_dict.items():
        #     video_name+=f'{k}_{v}_'
        video_name=video_name+f"_cost_{cost_int}"
        video_name = video_name + ".mp4"
        video_path = os.path.join(episode_dir, video_name)
        write_video(video_path, images, fps=5)


    

    return predicted_actions, 

def convert_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return [convert_to_list(item) for item in obj]
    else:
        return obj


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    all_predicted_actions = []




    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        actions=run_maniskill2_eval_single_episode(obj_episode_id=obj_episode_id, **kwargs)
                        all_predicted_actions.append(actions)


    converted_data = convert_to_list(all_predicted_actions)
    actions_file_path = os.path.join(args.logging_dir, f'{args.env_name}_steps_{args.max_episode_steps}.json')
    # with open(actions_file_path, 'w') as f:
    #     json.dump(converted_data, f)

    return all_predicted_actions
