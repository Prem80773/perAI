from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def load_captioner():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_stylish_caption(image, processor, model):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption
