import numpy as np
import colorsys
from scipy.spatial.distance import cosine
from config import *

def extract_color_histogram(image, bbox):
    from PIL import Image
    import cv2
    x1, y1, x2, y2 = bbox
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros(64)
    patch_rgb = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    patch_array = np.array(patch_rgb)
    hsv_patch = np.zeros_like(patch_array)
    for i in range(patch_array.shape[0]):
        for j in range(patch_array.shape[1]):
            r, g, b = patch_array[i, j] / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            hsv_patch[i, j] = [h * 179, s * 255, v * 255]
    h_hist = np.histogram(hsv_patch[:, :, 0], bins=16, range=(0, 180))[0]
    s_hist = np.histogram(hsv_patch[:, :, 1], bins=16, range=(0, 256))[0]
    v_hist = np.histogram(hsv_patch[:, :, 2], bins=16, range=(0, 256))[0]
    combined = np.concatenate([h_hist, s_hist, v_hist, np.histogram(hsv_patch.flatten(), bins=16)[0]])
    combined = combined.astype(np.float32)
    return combined / combined.sum() if combined.sum() > 0 else combined

def calculate_similarity(track, color_hist, bbox, width, height):
    color_sim = 1.0 - cosine(track.color_hist, color_hist)
    color_sim = max(0, color_sim)
    spatial_sim = calculate_spatial_consistency(track, bbox, width, height)
    return COLOR_WEIGHT * color_sim + SPATIAL_WEIGHT * spatial_sim

def calculate_spatial_consistency(track, bbox, width, height):
    if not track.exit_position:
        return 0.0
    x1, y1, x2, y2 = bbox
    if track.exit_position == 'left' and x1 <= EDGE_THRESHOLD:
        return 1.0
    if track.exit_position == 'right' and x2 >= width - EDGE_THRESHOLD:
        return 1.0
    if track.exit_position == 'top' and y1 <= EDGE_THRESHOLD:
        return 1.0
    if track.exit_position == 'bottom' and y2 >= height - EDGE_THRESHOLD:
        return 1.0
    return 0.0

def get_exit_position(bbox, width, height):
    x1, y1, x2, y2 = bbox
    if x1 <= EDGE_THRESHOLD:
        return 'left'
    elif x2 >= width - EDGE_THRESHOLD:
        return 'right'
    elif y1 <= EDGE_THRESHOLD:
        return 'top'
    elif y2 >= height - EDGE_THRESHOLD:
        return 'bottom'
    return None

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0
