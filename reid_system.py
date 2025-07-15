import cv2
import numpy as np
import time
import json
from tqdm import tqdm
from ultralytics import YOLO
from track import Track
from utils import (
    extract_color_histogram, calculate_similarity,
    calculate_iou, get_exit_position
)
from config import *

class PlayerReIDSystem:
    def __init__(self, model_path, conf_threshold, max_disappeared, stabilization_frames):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.max_disappeared = max_disappeared
        self.stabilization_frames = stabilization_frames

        self.tracks = {}
        self.next_id = 0
        self.frame_count = 0
        self.frame_width = 0
        self.frame_height = 0

        self.results_log = []
        self.frame_results = {}

    def update_tracks(self, detections, image):
        if not detections:
            for track in self.tracks.values():
                track.last_seen = self.frame_count
            return

        detection_features = []
        for bbox, conf in detections:
            color_hist = extract_color_histogram(image, bbox)
            detection_features.append((bbox, conf, color_hist))

        active_tracks = [t for t in self.tracks.values()
                         if self.frame_count - t.last_seen <= self.max_disappeared]

        if not active_tracks:
            for bbox, conf, color_hist in detection_features:
                self.create_new_track(bbox, conf, color_hist)
            return

        cost_matrix = np.zeros((len(active_tracks), len(detection_features)))

        for i, track in enumerate(active_tracks):
            for j, (bbox, _, color_hist) in enumerate(detection_features):
                sim = calculate_similarity(track, color_hist, bbox, self.frame_width, self.frame_height)
                cost_matrix[i, j] = 1.0 - sim

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for row, col in zip(row_ind, col_ind):
            if cost_matrix[row, col] < (1.0 - REID_THRESHOLD):
                track = active_tracks[row]
                bbox, conf, color_hist = detection_features[col]
                track.bbox = bbox
                track.color_hist = color_hist
                track.last_seen = self.frame_count
                track.confidence = conf
                track.frame_count += 1
                track.exit_position = None

                if track.frame_count >= self.stabilization_frames and not track.stabilized:
                    track.stabilized = True

                matched_tracks.add(track.id)
                matched_dets.add(col)

        for j, (bbox, conf, color_hist) in enumerate(detection_features):
            if j not in matched_dets:
                self.create_new_track(bbox, conf, color_hist)

        for track in self.tracks.values():
            if self.frame_count - track.last_seen == 1 and track.exit_position is None:
                track.exit_position = get_exit_position(track.bbox, self.frame_width, self.frame_height)

    def create_new_track(self, bbox, conf, color_hist):
        new_track = Track(
            id=self.next_id,
            bbox=bbox,
            color_hist=color_hist,
            last_seen=self.frame_count,
            confidence=conf,
            frame_count=1
        )
        self.tracks[self.next_id] = new_track
        self.next_id += 1

    def cleanup_tracks(self):
        to_remove = []
        for track_id, track in self.tracks.items():
            if self.frame_count - track.last_seen > self.max_disappeared:
                to_remove.append(track_id)
        for track_id in to_remove:
            del self.tracks[track_id]

    def get_stabilized_tracks(self):
        return {tid: t for tid, t in self.tracks.items()
                if t.stabilized and self.frame_count - t.last_seen <= 1}

    def process_frame(self, frame):
        self.frame_height, self.frame_width = frame.shape[:2]
        results = self.model(frame, conf=self.conf_threshold, classes=[0])
        detections = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                detections.append(((x1, y1, x2, y2), conf))

        self.update_tracks(detections, frame)
        self.cleanup_tracks()
        active_tracks = self.get_stabilized_tracks()
        self.frame_results[self.frame_count] = {
            'detections': len(detections),
            'active_tracks': len(active_tracks),
            'track_ids': list(active_tracks.keys())
        }
        self.frame_count += 1
        return active_tracks

    def process_video(self, video_path, output_path=None, show_display=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        start_time = time.time()
        frame_times = []

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t1 = time.time()
                active_tracks = self.process_frame(frame)
                t2 = time.time()

                display = frame.copy()
                for tid, track in active_tracks.items():
                    x1, y1, x2, y2 = track.bbox
                    color = (0, 255, 0) if track.stabilized else (0, 255, 255)
                    label = f"ID: {tid}" + (" (pending)" if not track.stabilized else "")
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.putText(display, f"Frame: {self.frame_count-1} | Active: {len(active_tracks)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if show_display:
                    cv2.imshow("Player Re-ID", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if output_path:
                    out.write(display)

                frame_times.append(t2 - t1)
                pbar.update(1)

        cap.release()
        if output_path:
            out.release()
        if show_display:
            cv2.destroyAllWindows()

        total_time = time.time() - start_time
        return {
            'video_info': {
                'path': video_path,
                'resolution': f"{width}x{height}",
                'fps': fps,
                'total_frames': total_frames
            },
            'processing_stats': {
                'avg_fps': len(frame_times) / total_time,
                'avg_frame_time': np.mean(frame_times),
                'total_time': total_time
            },
            'tracking_stats': {
                'max_simultaneous_tracks': max([len(f['track_ids']) for f in self.frame_results.values()], default=0),
                'total_unique_ids': self.next_id,
                'stabilization_frames': self.stabilization_frames
            },
            'frame_results': self.frame_results
        }

    def save_results(self, results, output_file):
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_file}")
