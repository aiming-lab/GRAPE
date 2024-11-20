import base64
import re
import cv2
import numpy as np
from openai import OpenAI
from matplotlib import cm
from gradio_client import Client, handle_file
import tempfile
class TrajectoryStageGenerator:
    def __init__(self, base_url, api_key, model_name):

        self.client = Client("")


        # self.client = OpenAI(
        #     base_url=base_url,
        #     api_key=api_key,
        # )
        # self.model_name = model_name

    @staticmethod
    def extract_points(response_text):
        point_pattern = r"\((\d+\.\d+),\s*(\d+\.\d+)\)"
        matches = re.findall(point_pattern, response_text)
        points = [(float(x), float(y)) for x, y in matches]
        return points

    @staticmethod
    def draw_lines_on_image_cv(image, points, num_subdivisions=100):
        height, width, _ = image.shape


        pixel_points = []
        for point in points:
            x = int(point[0] * width)
            y = int(point[1] * height)
            pixel_points.append((x, y))

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  
        font_thickness = 1
        text_color = (255, 255, 255) 

        for idx, (x, y) in enumerate(pixel_points):
            text = str(idx)
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_x = x - text_size[0] // 2  # Center the text horizontally
            text_y = y - 10  # Position text above the point
            cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            # Optionally, draw a circle to mark the point
            cv2.circle(image, (x, y), radius=3, color=(255, 255, 255), thickness=-1)


        pixel_points = np.array(pixel_points, dtype=np.float32)

 
        distances = [0]
        for i in range(1, len(pixel_points)):
            dist = np.linalg.norm(pixel_points[i] - pixel_points[i - 1])
            distances.append(distances[-1] + dist)
        total_distance = distances[-1]

   
        num_samples = num_subdivisions
        sample_distances = np.linspace(0, total_distance, num_samples)


        interpolated_points = []
        idx = 0
        for sd in sample_distances:
        
            while idx < len(distances) - 2 and sd > distances[idx + 1]:
                idx += 1
      
            t = (sd - distances[idx]) / (distances[idx + 1] - distances[idx])
            point = (1 - t) * pixel_points[idx] + t * pixel_points[idx + 1]
            interpolated_points.append(point)
        interpolated_points = np.array(interpolated_points, dtype=np.int32)


        cmap = cm.get_cmap('jet')
        colors = (cmap(np.linspace(0, 1, len(interpolated_points)))[:, :3] * 255).astype(np.uint8)  


        for i in range(len(interpolated_points) - 1):
            pt1 = tuple(interpolated_points[i])
            pt2 = tuple(interpolated_points[i + 1])
            color = tuple(int(c) for c in colors[i])
            cv2.line(image, pt1, pt2, color=color, thickness=2)

        return image

    # def process_image_and_quest(self, image, quest):
    #     # Encode the image as base64
    #     image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR if needed
    #     _, encoded_image_array = cv2.imencode('.jpg', image_bgr)
    #     encoded_image = base64.b64encode(encoded_image_array.tobytes()).decode('utf-8')

    #     # Make the API request
    #     response = self.client.chat.completions.create(
    #         messages=[
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {
    #                         "type": "image_url",
    #                         "image_url": {
    #                             "url": f"data:image/jpeg;base64,{encoded_image}",
    #                         },
    #                     },
    #                     {
    #                         "type": "text",
    #                         "text": (
    #                             f"In the image, please execute the command described in <quest>{quest}</quest>.\n"
    #                             "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal.\n"
    #                             "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example:\n"
    #                             "<ans>[(0.25, 0.32), (0.32, 0.17), (0.13, 0.24), <action>Open Gripper</action>, (0.74, 0.21), <action>Close Gripper</action>, ...]</ans>\n"
    #                             "The tuple denotes point x and y location of the end effector of the gripper in the image. The action tags indicate the gripper action.\n"
    #                             "The coordinates should be floats ranging between 0 and 1, indicating the relative locations of the points in the image.\n"
    #                             "Make sure to include <ans> and </ans> tags in your response. Think it step by step."
    #                         ),
    #                     },
    #                 ],
    #             }
    #         ],            # result = self.client.predict(
            #     image=handle_file(tmpfile.name),
            #     quest=quest,
            #     max_tokens=512,
            #     temperature=0,
            #     top_p=0.95,
            #     top_left_str="0,0",
            #     bottom_right_str="360,360",
            #     api_name="/predict"
            # )lf.draw_lines_on_image_cv(image.copy(), points)

    #     return output_image, response_text


    #########gardio test #################
    def process_image_and_quest(self, image, quest):

        with tempfile.NamedTemporaryFile(suffix='.png') as tmpfile:
            cv2.imwrite(tmpfile.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            tmpfile.flush()


            result = self.client.predict(
                    image=handle_file(tmpfile.name),
                    quest=quest,
                    max_tokens=512,
                    temperature=0,
                    top_p=0.95,
                    top_left_str="0,0",
                    bottom_right_str="360,360",
                    api_name="/predict"
            )



        response_text = result
        image_path = result[0]
        image = cv2.imread(image_path)

        response_text = result[1]
        ans_content_match = re.search(r'<ans>(.*?)</ans>', response_text, re.DOTALL)
        if ans_content_match:
            ans_content = ans_content_match.group(1)
        else:
            print("Failed to extract content between <ans> and </ans>.")
            ans_content = ''


        pattern = r'\(([\d\.]+),\s*([\d\.]+)\)|<action>(.*?)</action>'
        matches = re.findall(pattern, ans_content)
        steps = []
        for match in matches:
            x, y, action = match
            if x and y:
                steps.append(('point', (float(x), float(y))))
            elif action:
                steps.append(('action', action.strip()))

        coordinate_matches = re.findall(r'\(([\d\.]+),\s*([\d\.]+)\)', response_text)
        coordinates = [(float(x), float(y)) for x, y in coordinate_matches]
        # output_image = self.draw_lines_on_image_cv(image.copy(), points)
        cv2.imwrite('output_image.png', image)

        return image,coordinates



if __name__ == '__main__':


    # generator = TrajectoryStageGenerator(
    #     base_url="",
    #     api_key="",
    #     model_name=""     # Replace with your model name
    # )


    # image = cv2.imread('/home/joel/projects/kaiyuan/GRAPE/dataset/4-1/0.png')
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # quest = "Pick up the grapes and put it in the black bowl."


    # output_image, response_text = generator.process_image_and_quest(image_rgb, quest)
    # print(response_text)
