import os
import subprocess
from config import Config

def extract_clip_and_frame(video_path, event_minute, event_second, 
                          duration_sec=Config.DURATION_SEC,
                          output_clip_dir=Config.output_clip_dir,
                          output_frame_dir=Config.output_dir,
                          clip_name=Config.output_name):
    
    os.makedirs(output_clip_dir, exist_ok=True)
    os.makedirs(output_frame_dir, exist_ok=True)
    
    clip_start_minute, clip_start_second, target_frame_second = Config.calculate_clip_times(
        event_minute, event_second
    )
    
    print(f"Evento en: {event_minute}:{event_second:02d}")
    print(f"Clip desde: {clip_start_minute}:{clip_start_second:02d} por {duration_sec} segundos")
    print(f"Frame objetivo en segundo {target_frame_second} del clip")
    
    clip_path = extract_clip(
        video_path=video_path,
        start_minute=clip_start_minute,
        start_second=clip_start_second,
        duration_sec=duration_sec,
        output_dir=output_clip_dir,
        output_name=clip_name
    )
    
    if clip_path is None:
        return None, None
    
    frame_path = extract_frame_from_clip(
        clip_path=clip_path,
        target_second=target_frame_second,
        output_dir=output_frame_dir,
        frame_name="extracted_frame.jpg"
    )
    
    return clip_path, frame_path

def extract_clip(video_path, start_minute, start_second, duration_sec, 
                output_dir, output_name):
    output_path = os.path.join(output_dir, output_name)
    
    start_time = f"00:{int(start_minute):02d}:{int(start_second):02d}"
    
    cmd = [
        'ffmpeg',
        '-i', video_path,       
        '-ss', start_time,        
        '-t', str(duration_sec),
        '-c:v', 'libx264',        
        '-c:a', 'aac',
        '-avoid_negative_ts', 'make_zero',
        '-y',
        output_path
    ]
    
    try:
        print(f"Ejecutando comando FFmpeg: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print(f"Clip extraído exitosamente: {output_path}")
            verify_clip_duration(output_path, duration_sec)
            return output_path
        else:
            print(f"Error de FFmpeg: {result.stderr}")
            return None
    except FileNotFoundError:
        print("FFmpeg no encontrado. Instala FFmpeg para usar esta función.")
        return None

def verify_clip_duration(clip_path, expected_duration):
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'csv=p=0',
        clip_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            actual_duration = float(result.stdout.strip())
            print(f"Duración esperada: {expected_duration}s")
            print(f"Duración real: {actual_duration:.2f}s")
            if abs(actual_duration - expected_duration) > 1:
                print(f"ADVERTENCIA: Diferencia significativa en duración!")
        else:
            print("No se pudo verificar la duración del clip")
    except Exception as e:
        print(f"Error verificando duración: {e}")

def extract_frame_from_clip(clip_path, target_second, output_dir, frame_name):
    output_path = os.path.join(output_dir, frame_name)
    
    cmd = [
        'ffmpeg',
        '-i', clip_path,
        '-ss', str(target_second),
        '-vframes', '1',
        '-q:v', '2',  # Alta calidad para el frame
        '-y',  # Sobrescribir archivo existente
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print(f"Frame extraído exitosamente: {output_path}")
            return output_path
        else:
            print(f"Error extrayendo frame: {result.stderr}")
            return None
    except FileNotFoundError:
        print("FFmpeg no encontrado.")
        return None