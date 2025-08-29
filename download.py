from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# OWLv2 ensemble
print("⏬ Descargando OWLv2 Ensemble...")
Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# OWLv2 base
print("⏬ Descargando OWLv2 Base...")
Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")

# OWL-ViT base
print("⏬ Descargando OWL-ViT Base...")
OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")

print("Modelos descargados y guardados en ~/.cache/huggingface/hub/")
