class Config:
    minute = 4
    second = 26 ## 4segundos antes del objetivo
    DURATION_SEC = 15
    
    #texts = [["Man in a bike"]]
    #csv_output = "bounding_boxes_dataset_bike.csv"
    desface = 4
    texts = [["Black SUV"]]  
    csv_output = "bounding_boxes_dataset_car.csv"

    video_path = "/home/luis/Documents/FGV/Laboratory/entity-detection/videos/study_case_1_HD.mp4"
    output_name = "clip.mp4"
    output_clip_dir = "./clips"
    output_dir = "./frames"
    output_bbox_dir = "./bbox"
    model_id = "google/owlv2-base-patch16-ensemble"
    image_path = "./frames/extracted_frame.jpg"
    
    TARGET_FRAME_SECOND = None  
    TRACKING_CONFIDENCE_THRESHOLD = 0.3
    REDETECTION_INTERVAL = 15  # frames
    MAX_LOST_FRAMES = 30


    @classmethod
    def calculate_clip_times(cls, event_minute, event_second):
        total_event_seconds = event_minute * 60 + event_second
        
        clip_start_seconds = total_event_seconds
        
        clip_start_minute = clip_start_seconds // 60
        clip_start_second = clip_start_seconds % 60
        
        target_frame_second = 0 
        
        return clip_start_minute, clip_start_second, target_frame_second
