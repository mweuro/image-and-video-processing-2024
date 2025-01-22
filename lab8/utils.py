import numpy as np
import matplotlib.pyplot as plt
import cv2


def extract_frames(file: str, frame_rate: int = 1) -> list[np.ndarray]:
    video = cv2.VideoCapture(file)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    for i in range(int(frame_count)):
        _, frame = video.read()
        if i % frame_rate == 0 and frame is not None:
            frames.append(frame[..., ::-1])
    video.release()
    return frames


def optical_flow(frame_1, frame_2, **kwargs):
    # convert frames to grayscale
    gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_RGB2GRAY)
    gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_RGB2GRAY)
    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(gray_1, gray_2, None, **kwargs)
    # calculate magnitude and angle of 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # magnitude_normalized = magnitude_normalized.astype(np.uint8)
    return magnitude.astype(np.uint8), angle