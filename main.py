from tracking import VideoTracker
from clip import extract_clip
from config import Config
from data import create_dataset_from_tracking
import cv2
import os 

def main():
    clip_path = extract_clip(video_path=Config.video_path,
                             start_minute=Config.minute,
                             start_second=Config.second,
                             duration_sec=Config.DURATION_SEC,
                             output_dir=Config.output_clip_dir,
                             output_name=Config.output_name)
    
    if not clip_path:
        return
    print(f"Clip extraído: {clip_path}")
    
    tracker = VideoTracker()
    
    output_video_path = os.path.join(Config.output_clip_dir, "tracked_" + Config.output_name)

    cap = cv2.VideoCapture(clip_path)
    video_info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS))
    }
    cap.release()
    
    tracking_results = tracker.process_video(
        clip_path, 
        output_video_path,
    )

    dataset_dir = os.path.join(Config.output_clip_dir, "dataset")
    csv_path, dataframe = create_dataset_from_tracking(
        tracking_results, 
        dataset_dir, 
        csv_name=Config.csv_output, 
        video_info=video_info
    )
        
    return tracking_results, csv_path, dataframe
    

if __name__ == "__main__":
    main()