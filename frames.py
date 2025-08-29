import cv2
import os
from config import Config
    
def get_frame(video_path=Config.video_path, minute=0, second=0, output_dir=Config.output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video")

    #fps = 60
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int((minute * 60 + second) * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(f"No se pudo leer el frame")
    
    output_path = os.path.join(output_dir, "extracted_frame.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Frame guardado en {output_path}")

    cap.release()
    return frame

#frame = get_frame(minute=Config.minute, second=Config.second)