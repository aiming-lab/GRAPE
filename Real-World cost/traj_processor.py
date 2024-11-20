import numpy as np
import os
from glob import glob
import cv2
from scipy.spatial.transform import Rotation as R
import pickle
from PIL import Image
from tqdm import tqdm
from shapely.geometry import LineString
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import yaml

class TrajectoryProcessor:
    def __init__(self, config_file='../config/traj_config.yaml'):
        # Read config file
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Read config parameters
        self.cam_info_filepath = config['cam_info_filepath']
        self.distortion = np.array(config['distortion'])

        self.GRIPPER_CLOSE_THRESH = config['gripper']['close_thresh']
        self.GRIPPER_OPEN = config['gripper']['open_value']
        self.GRIPPER_CLOSE = config['gripper']['close_value']

        self.draw_point_only = config['draw_point_only']

        self.num_subdivisions = config.get('num_subdivisions', 100)
        self.tolerance = config.get('tolerance', 0.05)
        self.noise_level = config.get('noise_level', 0.01)

        self.dataset_root = config['dataset_root']

        # Read camera info
        with open(self.cam_info_filepath, 'rb') as f:
            cam_info = pickle.load(f)
            RAW_INTR = (cam_info['raw_intrinsics'][0, 0], cam_info['raw_intrinsics'][1, 1],
                        cam_info['raw_intrinsics'][0, 2], cam_info['raw_intrinsics'][1, 2])
            Fx, Fy, Cx, Cy = RAW_INTR
            x_min, x_max, y_min, y_max = cam_info["camera_crop"]
            INTR = (Fx, Fy, Cx - y_min, Cy - x_min)
            # 360x360 intrinsics
            self.intrinsic = np.array([
                [Fx, 0, Cx - y_min],
                [0, Fy, Cy - x_min],
                [0, 0, 1]
            ])
            self.extrinsic = cam_info['extrinsics']
            self.cam_crop = cam_info['camera_crop']

    def draw_lines_on_image_cv(self, image, points, draw_action=False, draw_index=False, thickness=2, full_traj=False):
        height, width, _ = image.shape

        pixel_points = []
        gripper_status = []
        for point in points:
            x = int(point[0] * width)
            y = int(point[1] * height)
            action = int(point[2])
            pixel_points.append((x, y))
            gripper_status.append(action)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  #
        font_thickness = 1
        text_color = (255, 255, 255)

        for idx, (x, y) in enumerate(pixel_points):
            if draw_index:
                text = str(idx)
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = x - text_size[0] // 2
                text_y = y - 10
                cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            if draw_action:
                if not full_traj:
                    if idx == 0 or gripper_status[idx] != gripper_status[idx - 1]:
                        circle_color = (0, 255, 0) if gripper_status[idx] == self.GRIPPER_CLOSE else (0, 0, 255)  # Green for close, red for open
                        cv2.circle(image, (x, y), 7, circle_color, thickness=thickness)
                else:
                    if idx != 0 and gripper_status[idx] != gripper_status[idx - 1]:
                        circle_color = (0, 255, 0) if gripper_status[idx] == self.GRIPPER_CLOSE else (0, 0, 255)
                        cv2.circle(image, (x, y), 7, circle_color, thickness=thickness)

        pixel_points = np.array(pixel_points, dtype=np.int32)

        for i in range(len(pixel_points) - 1):
            pt1 = tuple(pixel_points[i])
            pt2 = tuple(pixel_points[i + 1])
            color = (0, 0, 255)
            cv2.line(image, pt1, pt2, color=color, thickness=thickness)

        return image

    def draw_points_on_image_cv(self, image, points, full_traj, thickness=2):
        height, width, _ = image.shape

        # Convert normalized coordinates to pixel coordinates
        pixel_points = []
        gripper_status = []
        for point in points:
            x = int(point[0] * width)
            y = int(point[1] * height)
            action = int(point[2])
            pixel_points.append((x, y))
            gripper_status.append(action)

        # Draw initial position
        cv2.circle(image, (pixel_points[0][0], pixel_points[0][1]), 7, (0, 0, 0), thickness=thickness)

        for idx, (x, y) in enumerate(pixel_points):
            if not full_traj:
                if idx == 0 or gripper_status[idx] != gripper_status[idx - 1]:
                    circle_color = (0, 255, 0) if gripper_status[idx] == self.GRIPPER_CLOSE else (0, 0, 255)
                    cv2.circle(image, (x, y), 7, circle_color, thickness=thickness)
            else:
                if idx != 0 and gripper_status[idx] != gripper_status[idx - 1]:
                    circle_color = (0, 255, 0) if gripper_status[idx] == self.GRIPPER_CLOSE else (0, 0, 255)
                    cv2.circle(image, (x, y), 7, circle_color, thickness=thickness)
        return image

    def project_gripper_to_2d(self, episode, step_idx):
        gripper_pose_in_wrist = np.eye(4)
        gripper_pose_in_wrist[:3, 3] = np.array([0, 0, 0.14])

        # Extract end-effector pose from episode
        keypose_extrinsic = np.eye(4)
        keypose_extrinsic[:3, :3] = R.from_euler("xyz", episode[step_idx]["lowdim_ee"][3:6]).as_matrix()
        keypose_extrinsic[:3, 3] = episode[step_idx]["lowdim_ee"][:3]

        gripper_in_world = keypose_extrinsic @ gripper_pose_in_wrist
        intrinsic_camera = self.intrinsic

        def project_3d_to_2d_cv2(point_3d, intrinsic, extrinsic, distortion):
            robot_pose_homogeneous = np.append(point_3d, 1)
            extrinsic_inv = np.linalg.inv(extrinsic)
            point_in_camera_frame = extrinsic_inv @ robot_pose_homogeneous
            point_in_camera_frame = point_in_camera_frame[:3]
            point_in_camera_frame = point_in_camera_frame.reshape(1, 1, 3)
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
            projected_point_2d, _ = cv2.projectPoints(point_in_camera_frame, rvec, tvec, intrinsic, distortion)
            return projected_point_2d

        gripper_3d_position = gripper_in_world[:3, 3]
        projected_gripper_2d = project_3d_to_2d_cv2(gripper_3d_position, intrinsic_camera, self.extrinsic, self.distortion)
        return projected_gripper_2d.squeeze()

    def get_keyframes(self, trajectory_2d, gripper_action):
        action_index = []
        current_state = self.GRIPPER_OPEN
        for index in range(2, len(trajectory_2d)):
            if current_state == self.GRIPPER_OPEN and gripper_action[index - 2] == self.GRIPPER_OPEN \
               and gripper_action[index - 1] == self.GRIPPER_OPEN and gripper_action[index] == self.GRIPPER_CLOSE:
                action_index.append(index)
                current_state = self.GRIPPER_CLOSE
            if current_state == self.GRIPPER_CLOSE and gripper_action[index - 2] == self.GRIPPER_CLOSE \
               and gripper_action[index - 1] == self.GRIPPER_CLOSE and gripper_action[index] == self.GRIPPER_OPEN:
                action_index.append(index)
                current_state = self.GRIPPER_OPEN
        if len(action_index) > 0 and gripper_action[action_index[-1]] == self.GRIPPER_CLOSE:
            action_index.append(len(trajectory_2d) - 1)
        return action_index

    def get_sketch_v5_uniform_kernel(self, trajectory_2d, gripper_action):
        sketch_points = []
        sketch_point_str = []
        action_index = self.get_keyframes(trajectory_2d, gripper_action)
        if len(action_index) <= 1:
            return [], [], []

        sketch_points.append([trajectory_2d[action_index[0]][0], trajectory_2d[action_index[0]][1], self.GRIPPER_CLOSE])
        sketch_point_str.append(f"({trajectory_2d[action_index[0]][0]:.2f}, {trajectory_2d[action_index[0]][1]:.2f})")
        sketch_point_str.append("<action>Close Gripper</action>")
        current_gripper_status = self.GRIPPER_CLOSE
        for index in range(1, len(action_index)):

            start_index = action_index[index - 1]
            end_index = action_index[index] + 1
            subline = trajectory_2d[start_index:end_index]
            subline = [[x, y] for x, y in subline]
            simple_subline = subline

            for point in simple_subline[1:]:
                point_str = f"({point[0]:.2f}, {point[1]:.2f})"
                sketch_points.append([point[0], point[1], current_gripper_status])
                sketch_point_str.append(point_str)
            if gripper_action[action_index[index]] == self.GRIPPER_CLOSE:
                sketch_point_str.append("<action>Close Gripper</action>")
                current_gripper_status = self.GRIPPER_CLOSE
            else:
                sketch_point_str.append("<action>Open Gripper</action>")
                current_gripper_status = self.GRIPPER_OPEN
            sketch_points[-1][-1] = current_gripper_status

        return sketch_point_str, sketch_points, action_index

    def process_points(self, draw_action=True, full_traj=True, draw_point_only=None):
        if draw_point_only is None:
            draw_point_only = self.draw_point_only

        file_dirs = sorted(glob(f'{self.dataset_root}/episode_*.npy'))
        print(f"{len(file_dirs)} episodes found")

        for file_dir in tqdm(file_dirs):
            print("Processing", file_dir)

            episode = np.load(file_dir, allow_pickle=True)
            first_frame = episode[0]['215122255213_rgb']
            points = [self.project_gripper_to_2d(episode, i) for i in range(len(episode))]
            points_normed = [[x[0] / (first_frame.shape[1]), x[1] / (first_frame.shape[0])] for x in points]
            gripper_status = [1 - episode[i]['action'][-1] for i in range(len(episode))]
            gripper_status = [1 if gripper < self.GRIPPER_CLOSE_THRESH else 0 for gripper in gripper_status]

            _, trajectory_2d, _ = self.get_sketch_v5_uniform_kernel(points_normed, gripper_status)
            if full_traj:
                trajectory_2d.insert(0, [points_normed[0][0], points_normed[0][1], gripper_status[0]])
            trajectory_2d = np.array(trajectory_2d)

            if draw_point_only:
                traj_folder_name = "kyle_key_point/"
                img = self.draw_points_on_image_cv(first_frame, trajectory_2d, full_traj=full_traj)
            else:
                traj_folder_name = "kyle_traj/"
                img = self.draw_lines_on_image_cv(first_frame, trajectory_2d, draw_action=draw_action, full_traj=full_traj)

            save_filepath = file_dir.replace("npy/", traj_folder_name).replace(".npy", ".png")
            os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
            Image.fromarray(img).save(save_filepath)
            print("Saved", save_filepath)

    def find_nearest_indices(self, points_normed, stage_points):
        points_normed = np.array(points_normed)
        stage_points = np.array(stage_points)

        distances = np.sqrt(((stage_points[:, np.newaxis, :] - points_normed) ** 2).sum(axis=2))

        nearest_indices = np.argmin(distances, axis=1)

        return nearest_indices

    def process_stage_split(self, stage_points):
        stage_index_list=[]
        file_dirs = sorted(glob(f'{self.dataset_root}/episode_*.npy'))
        print(f"{len(file_dirs)} episodes found")

        for file_dir in tqdm(file_dirs):
            print("Processing", file_dir)

            episode = np.load(file_dir, allow_pickle=True)
            first_frame = episode[0]['215122255213_rgb']
            points = [self.project_gripper_to_2d(episode, i) for i in range(len(episode))]
            points_normed = [[x[0] / (first_frame.shape[1]), x[1] / (first_frame.shape[0])] for x in points]
            gripper_status = [1 - episode[i]['action'][-1] for i in range(len(episode))]
            gripper_status = [1 if gripper < self.GRIPPER_CLOSE_THRESH else 0 for gripper in gripper_status]

            stage_index = self.find_nearest_indices(points_normed, stage_points)
            stage_index_list.append(stage_index)


        return stage_index_list


if __name__ == '__main__':
    # processor = TrajectoryProcessor()

    # # To process using process_uw_points()
    # processor.process_points()

    # stage_points = [[0.5, 0.5], [0.6, 0.6]]
    # stage=processor.process_stage_split(stage_points)
    # print(stage)
