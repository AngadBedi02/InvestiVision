import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io
import torch  # YOLOv5 for object detection

# Streamlit custom styling (CSS)
st.markdown("""
    <style>
    body {
        background-color: #f4f4f9;  /* Light background color */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main .block-container {
        background-color: #ffffff; /* White background for main area */
        border-radius: 12px;
        padding: 20px;
    }
    h1 {
        color: #333333;
    }
    h2, h3, h4 {
        color: #444444;
    }
    .sidebar .sidebar-content {
        background-color: #2b2d42;  /* Dark sidebar */
        color: #f1f1f1;
    }
    .sidebar .sidebar-header {
        color: #ffffff;
    }
    .sidebar .sidebar-footer {
        background-color: #2b2d42;
    }
    .css-18e3th9 {
        background-color: #2b2d42;  /* Change button color */
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    .stImage {
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Clear Streamlit cache to avoid stale data
st.cache_data.clear()  # Clears data caches
st.cache_resource.clear()  # Clears resource caches

# Title and Instructions
st.title("InvestiVision")
st.write("Upload an image to apply various enhancements and perform object detection for better visibility of details.")
st.markdown("---")  # Adding a separator for clarity

# Load and process image functions
def load_image(image_file):
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def adjust_brightness_contrast(image, brightness=1.0, contrast=1.0):
    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness * 50)
    return adjusted

def reduce_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def edge_detection(image, low_threshold=100, high_threshold=200):
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening filter
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def grayscale_conversion(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return grayscale

def histogram_equalization(image):
    # Check if the image is already grayscale
    if len(image.shape) == 3:  # 3 channels (RGB)
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:  # The image is already grayscale
        grayscale = image

    # Apply histogram equalization
    equalized = cv2.equalizeHist(grayscale)
    return equalized

def convert_image(image):
    return Image.fromarray(image)

# Load YOLOv5 model for object detection
@st.cache_resource
def load_yolov5_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # Load YOLOv5 small model
    return model

# Detect objects in an image using YOLOv5
def detect_objects(image):
    model = load_yolov5_model()
    results = model(image)  # Pass image to model for prediction
    return results

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Store uploaded file in session state
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file

# Check if file is uploaded
if 'uploaded_file' in st.session_state:
    image = load_image(st.session_state.uploaded_file)

    # Display original image side-by-side with enhanced images
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Original Image', use_container_width=True)

    # Enhancement Controls
    brightness = st.sidebar.slider("Brightness", -5.0, 5.0, 1.0)
    contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)
    low_threshold = st.sidebar.slider("Edge Detection - Low Threshold", 50, 150, 100)
    high_threshold = st.sidebar.slider("Edge Detection - High Threshold", 150, 300, 200)

    # Additional enhancement options
    apply_sharp = st.sidebar.checkbox("Apply Sharpening")
    apply_grayscale = st.sidebar.checkbox("Convert to Grayscale")
    apply_hist_eq = st.sidebar.checkbox("Apply Histogram Equalization for Contrast")

    # Apply brightness and contrast adjustment
    adjusted_image = adjust_brightness_contrast(image, brightness, contrast)

    # Apply noise reduction
    noise_reduced_image = reduce_noise(adjusted_image)

    # Apply edge detection
    edges = edge_detection(noise_reduced_image, low_threshold, high_threshold)

    # Apply additional enhancements based on user selection
    enhanced_image = noise_reduced_image  # Start with noise reduced image as the base

    if apply_sharp:
        enhanced_image = apply_sharpening(enhanced_image)

    if apply_grayscale:
        enhanced_image = grayscale_conversion(enhanced_image)

    if apply_hist_eq:
        enhanced_image = histogram_equalization(enhanced_image)

    with col2:
        st.image(enhanced_image, caption="Enhanced Image", use_container_width=True)

    # Object detection with YOLOv5
    detection_results = detect_objects(image)

    # Display detection results
    st.subheader("Detected Objects")
    st.image(detection_results.render()[0], caption="Objects Detected", use_container_width=True)

    # Download functionality for enhanced image
    st.sidebar.header("Download Enhanced Image")
    if st.sidebar.button("Download"):
        final_image = convert_image(enhanced_image)
        buf = io.BytesIO()
        final_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label="Download Enhanced Image", data=byte_im, file_name="enhanced_image.png", mime="image/png")

else:
    st.write("Please upload an image to start applying enhancements.")
