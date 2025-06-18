import cv2
import numpy as np
from PIL import Image

# Load EDSR model once
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("weights/EDSR_x4.pb")
sr.setModel("edsr", 4)

def enhance_image(pil_image):
    # Convert PIL to OpenCV format (RGB ➝ BGR)
    img = np.array(pil_image)[:, :, ::-1]
    
    # Apply Super Resolution
    result = sr.upsample(img)

    # Convert back to RGB for PIL
    result_rgb = result[:, :, ::-1]
    return Image.fromarray(result_rgb)
