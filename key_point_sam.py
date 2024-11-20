import os
import cv2
import json
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision.ops import box_convert
from supervision.draw.color import ColorPalette
import supervision as sv
import pycocotools.mask as mask_util
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from GRAPE.Grounded_SAM_2.sam2.build_sam import build_sam2
from GRAPE.Grounded_SAM_2.sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.supervision_utils import CUSTOM_COLOR_MAP

class ObjectCenterDetector:
    def __init__(self, config_path):

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.grounding_model_id = config['GROUNDING_MODEL']
        self.text_prompt = config['TEXT_PROMPT']
        self.img_path = config['IMG_PATH']
        self.sam2_checkpoint = config['SAM2_CHECKPOINT']
        self.sam2_model_config = config['SAM2_MODEL_CONFIG']
        self.device = config.get('DEVICE')
        self.output_dir = Path(config.get('OUTPUT_DIR'))
        self.dump_json_results = config.get('DUMP_JSON_RESULTS')
        self.box_threshold = config.get('BOX_THRESHOLD')
        self.text_threshold = config.get('TEXT_THRESHOLD')

        self.output_dir.mkdir(parents=True, exist_ok=True)


        if self.device == 'cuda':
            torch.autocast(device_type=self.device, dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self.sam2_model = build_sam2(
            self.sam2_model_config,
            self.sam2_checkpoint,
            device=self.device
        )
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)


        self.processor = AutoProcessor.from_pretrained(self.grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_model_id).to(self.device)

    def detect_objects(self):

        image = Image.open(self.img_path).convert("RGB")
        self.sam2_predictor.set_image(np.array(image))


        inputs = self.processor(images=image, text=self.text_prompt.lower().strip(), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)


        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]]
        )


        if not results or len(results[0]["boxes"]) == 0:
            print("No objects detected.")
            return []

        input_boxes = results[0]["boxes"].cpu().numpy()
        confidences = results[0]["scores"].cpu().numpy().tolist()
        class_names = results[0]["labels"]
        class_ids = np.array(list(range(len(class_names))))


        with torch.cuda.amp.autocast(enabled=self.device == 'cuda'):
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

        # import pdb; pdb.set_trace()

 
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Visualize results using supervision
        img = cv2.imread(self.img_path)
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        # Annotate boxes and masks 
        box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

        # Save image
        output_image_path = self.output_dir / "grounded_sam2_annotated_image_with_mask.jpg"
        cv2.imwrite(str(output_image_path), annotated_frame)


        object_centers = []
        for idx, box in enumerate(input_boxes):
            x_min, y_min, x_max, y_max = box
            center_x = float((x_min + x_max) / 2)
            center_y = float((y_min + y_max) / 2)
            center_point = (center_x / image.width, center_y / image.height)
            object_info = {
                "class_name": class_names[idx],
                "center_point": center_point
            }

            object_centers.append(object_info)



        for obj in object_centers:
            obj['center_point'] = [float(coord) for coord in obj['center_point']]
            
            results_json = {
                "image_path": self.img_path,
                "object_centers": object_centers
            }

            output_json_path = self.output_dir / "keypoints_results.json"
            with open(output_json_path, 'w') as f:
                json.dump(results_json, f, indent=4)

        
        return object_centers

if __name__ == "__main__":

    # config_path = "../sam_config.yaml"

    # detector = ObjectCenterDetector(config_path=config_path)

    # # Detect objects and get their center points
    # object_centers = detector.detect_objects()
    # # import pdb; pdb.set_trace()

    # # Print the results
    # for obj in object_centers:
    #     print(f"Object: {obj['class_name']}, Center Point: {obj['center_point']}")

    
    # checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

    # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    #     image = Image.open('/home/joel/projects/kaiyuan/GRAPE/output_image.png').convert("RGB")
    #     predictor.set_image(image)
    #     masks, _, _ = predictor.predict()
    #     masks = masks.transpose(1, 2, 0)
    #     masks=masks[:,:,0]
    #     masks = (masks * 255).astype('uint8')
    #     # plt.imshow(masks)
    #     # plt.show()
    #     plt.imsave("zero_image.png", masks)
    #     import pdb; pdb.set_trace()
