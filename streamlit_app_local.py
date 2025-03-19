import streamlit as st
import cv2
import numpy as np
from preprocessing.image_processing import preprocess_image, detect_ridges, analyze_ridge_patterns
from features.minutiae_extraction import extract_minutiae, analyze_ridge_characteristics
from matching.matcher import match_fingerprints
import logging
import traceback
import tempfile
import os
from PIL import Image
import io
from datetime import datetime

# تكوين التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تكوين الصفحة
st.set_page_config(
    page_title="نظام تحليل البصمات",
    page_icon="🔍",
    layout="wide"
)

# العنوان الرئيسي
st.title("نظام تحليل البصمات الجنائي")
st.markdown("""
### نظام متقدم لتحليل ومطابقة البصمات باستخدام تقنيات الذكاء الاصطناعي
""")

# تعريف الدوال المساعدة
def validate_image(image):
    if image is None:
        return False
    if image.size == 0:
        return False
    if len(image.shape) != 2:
        return False
    return True

def display_image(image, caption):
    try:
        if isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                # If it's a color image
                image = Image.fromarray(image)
            else:
                # If it's a grayscale image
                image = Image.fromarray(image.astype(np.uint8))
        elif not isinstance(image, Image.Image):
            logger.error(f"Unsupported image type: {type(image)}")
            return False
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Display using Streamlit with use_container_width parameter
        st.image(image, caption=caption, use_container_width=True)
        return True
    except Exception as e:
        logger.error(f"Error displaying image: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# باقي الكود كما هو في الملف الأصلي 