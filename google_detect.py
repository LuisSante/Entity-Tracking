import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from config import Config
import os

class ObjectDetector:
    def __init__(self, model_id=Config.model_id):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {self.device}")
        
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device)
        print("Modelo cargado correctamente")

    def detect_objects(self, image_path, text_queries, threshold=0.3):
        if not os.path.exists(image_path):
            print(f"Imagen no encontrada: {image_path}")
            return None, None
            
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )
        
        return image, results

    def draw_detections(self, image, results, text_queries, threshold=0.3, batch_index=0):
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not results or len(results) == 0:
            print("No se encontraron resultados")
            return frame_bgr
        
        boxes = results[batch_index]["boxes"]
        scores = results[batch_index]["scores"]
        labels = results[batch_index]["labels"]
        
        detections_found = False
        
        for box, score, label in zip(boxes, scores, labels):
            if score >= threshold:
                detections_found = True
                x0, y0, x1, y1 = map(int, box.tolist())
                
                # Dibujar bounding box
                cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 255, 0), 2)
                
                # Dibujar etiqueta
                label_text = f"{text_queries[batch_index][label]} {score:.2f}"
                cv2.putText(
                    frame_bgr,
                    label_text,
                    (x0, max(y0 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                
                print(f"Detectado: {text_queries[batch_index][label]} con confianza {score:.3f} en {box.tolist()}")
        
        if not detections_found:
            print(f"No se encontraron objetos con confianza >= {threshold}")
        
        return frame_bgr

    def save_result(self, frame_bgr, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame_bgr)
        print(f"Resultado guardado en: {output_path}")

    def detect_and_visualize(self, image_path, text_queries, threshold=0.3, 
                           output_path="./bbox/owlv2_result.jpg"):
        image, results = self.detect_objects(image_path, text_queries, threshold)
        
        if image is None:
            return None
            
        frame_bgr = self.draw_detections(image, results, text_queries, threshold)
        self.save_result(frame_bgr, output_path)
        
        return results
    
#first =  ObjectDetector()
#result = first.detect_and_visualize(image_path=Config.image_path, 
#                          text_queries=Config.texts,
#                           threshold=0.3)