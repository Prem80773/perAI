import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from scorer import score_image  # custom scoring logic
from captioner import load_captioner, generate_stylish_caption

st.set_page_config(page_title="per.AI – Pick & Style Your Best Photo", layout="wide")

# Load CLIP model and processor
@st.cache_resource
def load_models():
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    caption_processor, caption_model = load_captioner()
    return clip_model, clip_processor, caption_processor, caption_model

clip_model, clip_processor, caption_processor, caption_model = load_models()

# Prompts to score images against
score_prompts = [
    "a perfect Instagram photo",
    "aesthetic lighting and clear face",
    "happy and expressive moment",
    "stylish and visually appealing"
]

st.title("📸 per.AI – Your Smart Post Assistant")
st.markdown("Upload 5–20 photos. We'll score and rank them using AI!")

uploaded_files = st.file_uploader("Upload your best shots", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) < 5 or len(uploaded_files) > 20:
        st.warning("Please upload between 5 and 20 photos.")
    else:
        st.subheader("🔍 Scoring Images...")
        scored_images = []

        for uploaded_file in uploaded_files:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                total_score = 0

                for prompt in score_prompts:
                    score = score_image(image, clip_model, clip_processor, prompt)
                    total_score += score

                avg_score = total_score / len(score_prompts)
                scored_images.append((uploaded_file.name, image, avg_score))

            except Exception as e:
                st.warning(f"⚠️ Failed to process {uploaded_file.name}: {str(e)}")

        if not scored_images:
            st.error("❌ No image was successfully scored.")
        else:
            # Show top 3
            top_images = sorted(scored_images, key=lambda x: x[2], reverse=True)[:3]
            st.subheader("🏆 Top 3 Photos Picked by AI")
            cols = st.columns(3)
            selected = None

            for i, (name, img, score) in enumerate(top_images):
                with cols[i]:
                    st.image(img, caption=f"{name}\nScore: {round(score, 4)}", use_column_width=True)
                    if st.button(f"Select This", key=f"select_{i}"):
                        selected = (name, img)

            if selected:
                st.subheader("✍️ Generating Your Custom Caption...")
                caption = generate_stylish_caption(selected[1], caption_processor, caption_model)

                st.image(selected[1], caption="Your Final Pick", use_column_width=True)
                st.success(f"📝 Caption: {caption}")
