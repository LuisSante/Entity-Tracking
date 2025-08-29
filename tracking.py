import cv2
import torch
import os

from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from motpy import Detection, MultiObjectTracker
from config import Config
import numpy as np

class VideoTracker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.processor = Owlv2Processor.from_pretrained(Config.model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(Config.model_id).to(self.device)
        
        self.tracker = MultiObjectTracker(dt=1/14.0, tracker_kwargs={'max_staleness': 2})

        
    def detect_objects_in_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        
        inputs = self.processor(text=Config.texts, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.3
        )
        
        detections = []
        i = 0  # batch index
        boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
        
        for box, score, label in zip(boxes, scores, labels):
            if score >= 0.3:
                x1, y1, x2, y2 = map(int, box.tolist())
                entity_type = Config.texts[i][label] 
                detections.append((x1, y1, x2, y2, float(score), entity_type))
        
        return detections
    
    def update_tracker(self, detections):
        motpy_detections = []
        entities = []
        for det in detections:
            x1, y1, x2, y2, conf, entity_type = det
            bbox = [x1, y1, x2-x1, y2-y1]
            detection = Detection(box=bbox, score=conf, feature=np.array([hash(entity_type)]))
            motpy_detections.append(detection)
            entities.append(entity_type)

        self.tracker.step(motpy_detections)

        tracks = []
        for idx, track in enumerate(self.tracker.active_tracks()):
            x, y, w, h = track.box
            tracks.append({
                'id': track.id,
                'bbox': [int(x), int(y), int(x+w), int(y+h)],
                'score': track.score,
                'entity_type': entities[idx] if idx < len(entities) else None
            })
        return tracks
    
    def process_video(self, video_path, output_path=None, skip_frames=10):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        out = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        all_tracks_history = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % skip_frames == 0:
                    print(f"Processing frame {frame_count + 1}/{total_frames}")
                    
                    detections = self.detect_objects_in_frame(frame)
                    tracks = self.update_tracker(detections)
                    print(f"Frame {frame_count + 1}: {len(detections)} detecciones, {len(tracks)} tracks activos")
                    
                    frame_tracks = {
                        'frame': frame_count,
                        'tracks': tracks.copy()
                    }
                    all_tracks_history.append(frame_tracks)
                    
                    frame_with_tracking = self.draw_tracking_results(frame, tracks)
                    
                    if out:
                        out.write(frame_with_tracking)
                    
                    cv2.imshow('Tracking', frame_with_tracking)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
                    processed_count += 1
                
                frame_count += 1
                
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
        
        print(f"Processed {frame_count} frames")
        return all_tracks_history
    
    def draw_tracking_results(self, frame, tracks):
        ENTITY_COLORS = {
            "Man in a bike": (0, 0, 255),   # Rojo
            "Black SUV": (0, 255, 0),       # Verde
        }
        
        frame_copy = frame.copy()
        
        for track in tracks:
            x1, y1, x2, y2 = track['bbox']
            score = track['score']
            entity = track.get('entity_type', 'Unknown')
            
            color = ENTITY_COLORS.get(entity, (255, 255, 255))
            if color == (255, 255, 255):
                continue
            
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            label = f"{entity} ({score:.2f})"
            cv2.putText(
                frame_copy,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        return frame_copy
    
    def process_target_frame(self, frame_path, output_bbox_dir):
        frame = cv2.imread(frame_path)
        if frame is None:
            print("Error: No se pudo leer el frame")
            return None, None

        detections = self.detect_objects_in_frame(frame)
        tracks = self.update_tracker(detections)

        frame_with_bbox = self.draw_tracking_results(frame, tracks)
        bbox_filename = os.path.join(output_bbox_dir, "extracted_frame_bbox.txt")
        with open(bbox_filename, 'w') as f:
            for track in tracks:
                x1, y1, x2, y2 = track['bbox']
                score = track['score']
                f.write(f"{x1},{y1},{x2},{y2},{score:.2f}\n")

        print(f"Bounding boxes guardados en: {bbox_filename}")
        return frame_with_bbox, tracks