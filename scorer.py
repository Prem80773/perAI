import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# This function scores how well the image matches the given prompt
def score_image(image: Image.Image, model: CLIPModel, processor: CLIPProcessor, prompt: str) -> float:
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: [1, 1]
        score = logits_per_image.item()  # extract float score
    return score
