import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as transforms
from PIL import Image

# 1. Config page and models
st.set_page_config(page_title="Real vs Fake Image Detector", layout="wide")
st.title("Real vs Fake Image Detector")

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


@st.cache_resource  # Cache model to prevent reloading when upload images
def load_model():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(
        torch.load(
            "./dire_detection/resnet50_best_real_fake.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.to(device).eval()
    return model


model = load_model()

# 2. Preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# 3. Upload UI
uploaded_files = st.file_uploader(
    "Drag and drop one/multiple images...",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    # Show result as grid column
    cols = st.columns(4)

    for idx, file in enumerate(uploaded_files):
        img = Image.open(file).convert("RGB")

        # Inference
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        is_fake = pred.item() == 1
        label = "FAKE" if is_fake else "REAL"
        color = "red" if is_fake else "green"
        score = confidence.item() * 100

        # Show in the UI
        with cols[idx % 4]:
            st.image(img, use_container_width=True)
            st.markdown(f"**Result: :{color}[{label}]**")
            st.caption(f"Confidence: {score:.2f}%")
