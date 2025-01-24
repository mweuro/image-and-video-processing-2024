import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from typing import Literal



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



def optical_flow(frame_1: np.ndarray, frame_2: np.ndarray, normalize: bool = False, **kwargs) -> tuple[np.ndarray]:
    # convert frames to grayscale
    gray_1 = cv2.cvtColor(frame_1, cv2.COLOR_RGB2GRAY)
    gray_2 = cv2.cvtColor(frame_2, cv2.COLOR_RGB2GRAY)
    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(gray_1, gray_2, None, **kwargs)
    # calculate magnitude and angle of 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # if normalize is True, then normalize magnitude values
    if normalize:  
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return magnitude.astype(np.uint8), angle



def rgb_map(size: int = 500):
    # grid setup
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    # cartesian -> polar
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    # HSV -> RGB
    H = (Theta + np.pi) / (2 * np.pi)
    S = np.clip(R, 0, 1)
    V = np.ones_like(R)
    HSV = np.stack((H, S, V), axis = -1)
    RGB = plt.cm.hsv(HSV[..., 0])[:, :, :3]
    return RGB



def display_rgb_map(ax: matplotlib.axes, size: int = 500) -> None:
    
    # init rgb map, and radius
    rgb = rgb_map(size)
    R = 5
    # plot color map + circle
    ax.imshow(rgb, extent = (-(R + 1), R + 1, -(R + 1), R + 1), origin = 'lower')
    circle = plt.Circle((0, 0), R, color = 'black', fill = False, lw=2)
    ax.add_artist(circle)
    ticks_radius = R + 0.5
    for angle in range(0, 360, 30):
        radians = np.radians(angle)
        x_tick = ticks_radius * np.cos(radians)
        y_tick = ticks_radius * np.sin(radians)
        ax.text(
            x_tick, y_tick, f"{angle}°",
            color='black', ha='center', va='center', fontsize=8
        )
    # axes setup
    ax.axhline(0, color = 'black', linewidth = 1)
    ax.axvline(0, color = 'black', linewidth = 1)
    ax.set_aspect('equal')
    ax.set_xlim(-R - 1, R + 1)
    ax.set_ylim(-R - 1, R + 1)
    ax.set_title("RGB circle")
    ax.axis('off')



def display_optical_flow_no_legend(ax: matplotlib.axes, 
                      magnitude: np.ndarray, 
                      angle: np.ndarray,
                      image: np.ndarray = None,
                      plot_type: Literal['vector', 'flow'] = 'vector',
                      gap: int = 16) -> None:
    # grid setup
    x, y = np.meshgrid(np.arange(magnitude.shape[1]), np.arange(magnitude.shape[0]))
    u = magnitude * np.cos(angle)
    v = magnitude * np.sin(angle)
    x_gap = x[::gap, ::gap]
    y_gap = y[::gap, ::gap]
    u_gap = u[::gap, ::gap]
    v_gap = v[::gap, ::gap]
    angle_gap = angle[::gap, ::gap]
    # vector color setup
    hue = 1 - ((angle_gap + np.pi) / (2 * np.pi) % 1)
    colors = plt.cm.hsv(hue.flatten())[:, :3]
    # flow color setup
    hsv = np.zeros((magnitude.shape[0], magnitude.shape[1], 3), dtype = np.uint8)
    hsv[..., 0] = ((1 - ((angle + np.pi) / (2 * np.pi) % 1)) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Konwersja HSV -> RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    if plot_type == 'vector':
        # vector field plot
        ax.quiver(x_gap, y_gap, u_gap, v_gap, angles = "xy", scale_units = "xy", scale = 1, color = colors)
        if image is not None:
            dark_factor = 0.3
            darkened_image = np.clip(image * dark_factor, 0, 255).astype(np.uint8)
            ax.imshow(darkened_image)
        else:
            ax.imshow(np.ones(magnitude.shape))
    elif plot_type == 'flow':
        ax.imshow(rgb)
    else:
        raise ValueError('Incorrect plot type!')
    ax.set_xlim(0, magnitude.shape[1])
    ax.set_ylim(magnitude.shape[0], 0)
    ax.set_title("Optical Flow")
    ax.axis('off')


def plot_optical_flow(magnitude: np.ndarray, 
                      angle: np.ndarray,
                      image: np.ndarray = None,
                      plot_type: Literal['vector', 'flow'] = 'vector',
                      gap: int = 16,
                      show_legend: bool = True) -> None:
    
    if show_legend:
        _, axs = plt.subplots(1, 2, figsize = (20, 6))
        
        if image is not None:
            size = image.shape[0]
        display_optical_flow_no_legend(axs[0], magnitude, angle, image, plot_type, gap)
        display_rgb_map(axs[1], size)
        plt.tight_layout()
    else:
        _, axs = plt.subplots(1, 1, figsize = (12, 6))
        display_optical_flow_no_legend(axs, magnitude, angle, image, plot_type, gap)
        
    plt.show()