import os
import time
import argparse

import cv2
import torch
import torch.nn as nn
import pyvirtualcam
import numpy as np
import mediapipe as mp
from PIL import Image
import warnings
import torch
from time import perf_counter
from models import TalkingAnimeLight, TalkingAnimeLightCached
from pose import get_pose
from utils import preprocessing_image, postprocessing_image
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--input', type=str, default='cam')
parser.add_argument('--character', type=str, default='0001')
parser.add_argument('--output_dir', type=str, default=f'dst')
parser.add_argument('--output_webcam', action='store_true')
args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_pose_final(pose: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    pose_vector = torch.empty(1, 3)
    mouth_eye_vector = torch.empty(1, 27)
    eye_l_h_temp = pose[0]
    eye_r_h_temp = pose[1]
    mouth_ratio = pose[2]
    eye_y_ratio = pose[3]
    eye_x_ratio = pose[4]
    x_angle = pose[5]
    y_angle = pose[6]
    z_angle = pose[7]

    mouth_eye_vector[0, 2] = eye_l_h_temp
    mouth_eye_vector[0, 3] = eye_r_h_temp
    mouth_eye_vector[0, 14] = mouth_ratio * 1.5
    mouth_eye_vector[0, 25] = eye_y_ratio
    mouth_eye_vector[0, 26] = eye_x_ratio

    pose_vector[0, 0] = (x_angle - 1.5) * 1.6
    pose_vector[0, 1] = y_angle * 2.0  # temp weight
    pose_vector[0, 2] = (z_angle + 1.5) * 2  # temp weight
    return pose_vector, mouth_eye_vector
    

@torch.no_grad()
def main():
    model = TalkingAnimeLightCached().to(device)
    model = model.eval()
    model = model
    img = Image.open(f"character/{args.character}.png").resize((256, 256))
    input_image = preprocessing_image(img).unsqueeze(0).to(device)

    if args.input == 'cam':
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret is None:
            raise RuntimeError("Can't find Camera")
    else:
        cap = cv2.VideoCapture(args.input)
        frame_count = 0
        os.makedirs(os.path.join('dst', args.character, args.output_dir), exist_ok=True)

    facemesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    if args.output_webcam:
        cam = pyvirtualcam.Camera(width=1280, height=720, fps=30)
        print(f'Using virtual camera: {cam.device}')

    pose_queue = []

    while cap.isOpened():
        start_time = perf_counter()
        
        ret, frame = cap.read()
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = facemesh.process(input_frame)

        if results.multi_face_landmarks is None:
            # face not detected
            continue

        facial_landmarks = results.multi_face_landmarks[0].landmark

        if args.debug:
            pose, debug_image = get_pose(facial_landmarks, frame)
        else:
            pose = get_pose(facial_landmarks)

        pose_vector, mouth_eye_vector = get_pose_final(pose)
        pose_vector = pose_vector.to(device)
        mouth_eye_vector = mouth_eye_vector.to(device)
        with torch.no_grad():
            output_image = model(input_image, mouth_eye_vector, pose_vector)[0, ...]

        if args.debug:
            output_frame = cv2.cvtColor(postprocessing_image(output_image.cpu()), cv2.COLOR_RGBA2BGR)
            resized_frame = cv2.resize(output_frame, (np.min(debug_image.shape[:2]), np.min(debug_image.shape[:2])))
            output_frame = np.concatenate([debug_image, resized_frame], axis=1)
            cv2.putText(output_frame, f'FPS: {1 / (perf_counter() - start_time):.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("frame", output_frame)
        if args.input != 'cam':
            cv2.imwrite(os.path.join('dst', args.character, args.output_dir, f'{frame_count:04d}.jpeg'))
            frame_count += 1
        if args.output_webcam:
            result_image = np.zeros([720, 1280, 3], dtype=np.uint8)
            result_image[720 - 512:, 1280 // 2 - 256:1280 // 2 + 256] = cv2.resize(
                cv2.cvtColor(postprocessing_image(output_image.cpu()), cv2.COLOR_RGBA2RGB), (512, 512))
            cam.send(result_image)
            cam.sleep_until_next_frame()
            
        torch.cuda.empty_cache()
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
