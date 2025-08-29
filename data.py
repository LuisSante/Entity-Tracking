import pandas as pd
from pathlib import Path

class BoundingBoxDatasetExtractor:
    def __init__(self):
        self.dataset = []
    
    def extract_from_tracking_results(self, tracking_results, video_info=None):
        dataset = []
        ENTITY_COLORS = {
            "Man in a bike": (0, 0, 255),   
            "Black SUV": (0, 255, 0),      
        }
        
        for frame_data in tracking_results:
            frame_number = frame_data['frame']
            tracks = frame_data['tracks']
            
            for track in tracks:
                row = {
                    'frame_number': frame_number,
                    'track_id': track['id'],
                    'x1': track['bbox'][0],
                    'y1': track['bbox'][1], 
                    'x2': track['bbox'][2],
                    'y2': track['bbox'][3],
                    'width': track['bbox'][2] - track['bbox'][0],
                    'height': track['bbox'][3] - track['bbox'][1],
                    'confidence_score': track['score'],
                    'center_x': (track['bbox'][0] + track['bbox'][2]) / 2,
                    'center_y': (track['bbox'][1] + track['bbox'][3]) / 2,
                    'area': (track['bbox'][2] - track['bbox'][0]) * (track['bbox'][3] - track['bbox'][1]),
                    'entity_type': track.get('entity_type', 'Unknown'),
                    'color': ENTITY_COLORS.get(track.get('entity_type', 'Unknown'), (255, 255, 255)) # Blanco default
                }
                
                if video_info:
                    fps = video_info.get('fps', None)
                    timestamp_sec = frame_number / fps if fps else None
                    
                    row.update({
                        'video_width': video_info.get('width', None),
                        'video_height': video_info.get('height', None),
                        'fps': fps,
                        'timestamp_sec': timestamp_sec,
                        'timestamp_hms': str(pd.to_timedelta(timestamp_sec, unit='s')) if timestamp_sec is not None else None
                    })
                
                dataset.append(row)
        
        self.dataset = dataset
        return dataset
    
    def to_pandas_dataframe(self):
        return pd.DataFrame(self.dataset)
    
    def to_csv(self, output_path, include_headers=True):
        df = self.to_pandas_dataframe()
        df.to_csv(output_path, index=False, header=include_headers)
        print(f"Dataset guardado en CSV: {output_path}")
        return output_path


def create_dataset_from_tracking(tracking_results, output_dir, csv_name="dataset.csv", video_info=None):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    extractor = BoundingBoxDatasetExtractor()
    extractor.extract_from_tracking_results(tracking_results, video_info)
    
    csv_path = Path(output_dir) / csv_name
    extractor.to_csv(csv_path)
    
    return str(csv_path), extractor.to_pandas_dataframe()
