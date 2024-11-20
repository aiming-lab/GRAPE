import numpy as np
import torch
import cv2
import openai
import yaml
from torchvision import transforms
from PIL import Image

from traj_stage_gen import TrajectoryStageGenerator
from traj_processor import TrajectoryProcessor
from key_point_sam import ObjectCenterDetector
from key_point_dino import KeypointProposer
from GRAPE.Grounded_SAM_2.sam2.sam2_image_predictor import SAM2ImagePredictor
from GRAPE.Grounded_SAM_2.sam2.build_sam import build_sam2


from openai import OpenAI
import base64
import os
import copy

def read_grape_prompt(file_path):
    with open(file_path, 'r') as file:
        grape_prompt = file.read()
    return grape_prompt

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def api_call_2(img_path, prompt, temperature=0):
    base64_image1 = encode_image(img_path)

    response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },

                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image1}",
                            "detail": "high"
                        },
                    },
                ],
                }
            ],
            temperature=temperature,
            max_tokens=1024,
        )
    
    return response


if __name__ == '__main__':
    config = load_config("../config/config.yaml")


    #############input your inital state and prompt###################
    image_path = '../dataset/grape/0.png'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    quest = "Pick up the grape and put it in the black bowl."
    #############input your inital statprint(stage_point_list)e and prompt###################


    # using the vlm model to get the stage points

    generator = TrajectoryStageGenerator(
        base_url=config['trajectory_stage_generator']['base_url'],
        api_key=config['trajectory_stage_generator']['api_key'],
        model_name=config['trajectory_stage_generator']['model_name']
    )

    output_image, stage_point_list = generator.process_image_and_quest(image_rgb, quest)
    print(stage_point_list)

    ##test
    # stage_point_list=config['trajectory_stage_generator']['stage_point_list']
    # process the trajectory to split the stages

    processor = TrajectoryProcessor(config_file=config['trajectory_processor']['config_file'])
    processor.process_points()

    stage_points = stage_point_list
    stage=processor.process_stage_split(stage_points)
    print(stage)


    ## keypoint_sam get the keypoints of the image

    sam_config_path = config['object_center_detector']['config_path']
    detector = ObjectCenterDetector(config_path=sam_config_path)

    keypoint_sam = detector.detect_objects()
    for obj in keypoint_sam:
        print(f"Object: {obj['class_name']}, Center Point: {obj['center_point']}")



    # Read GRAPE_PROMPT from a file
    grape_prompt_path = '../vlm/default/grape_prompt.txt'  
    GRAPE_PROMPT = read_grape_prompt(grape_prompt_path)
    # Insert the quest into the GRAPE_PROMPT
    query_prompt = GRAPE_PROMPT.format(quest=quest)
    img_path = image_path  # Use the image path you have




    try: 
    
        os.environ['OPENAI_API_KEY'] = ""
        client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        response = api_call_2(img_path, query_prompt)


        result = ''
        for choice in response.choices:
            result += choice.message.content

        print(result)


    except:
        raise Exception("No gpt API keys available")















