import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Depression Detection App", layout="wide")

# Load the pre-trained model and tokenizer
model_path = "D:\Project\depression_bert_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Sidebar for navigation
page = st.sidebar.selectbox("Select Page", ["Home", "Settings"])

# Function to set styles based on mode
def set_styles(mode):
    if mode == "Dark Mode":
        background_color = "#333333"
        text_color = "#FFFFFF"
        button_color = "transparent"  
        button_text_color = "#FFFFFF"  
        sidebar_background_color = "#1e1e1e"  
        sidebar_text_color = "#FFFFFF"  
    elif mode == "Light Mode":
        background_color = "#FFFFFF"
        text_color = "#000000"
        button_color = "transparent"  
        button_text_color = "#000000"  
        sidebar_background_color = "#f0f2f6" 
        sidebar_text_color = "#000000"  
    else:  # System Mode (fallback)
        background_color = "#f0f2f6"  
        text_color = "#000000"
        button_color = "transparent"  
        button_text_color = "#000000"  
        sidebar_background_color = "#f0f2f6"  
        sidebar_text_color = "#000000"  
    
    # Add CSS to apply styles
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {background_color};
        }}
        .stButton {{
            background-color: {button_color};
            border: 2px solid {button_text_color};
            color: {button_text_color};
        }}
        .css-1d391kg {{
            color: {text_color};
        }}
        .stTextInput {{
            color: {text_color};
            background-color: {background_color};
        }}
        .stMarkdown {{
            color: {text_color};
        }}
        .stSidebar {{
            background-color: {sidebar_background_color};
        }}
        .stSidebar .sidebar-content {{
            color: {sidebar_text_color};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# App Title
st.markdown("<h1 style='text-align: center;'>Depression Detection App</h1>", unsafe_allow_html=True)

# Handle navigation between Home and Settings
if page == "Home":
    # Home Section: 
    col1, col2 = st.columns([2, 1])  

    with col1:
        st.markdown(
            """
            <p style='font-style: italic; font-size: 18px;'>Welcome to the Depression Detection App! This app analyzes text to identify potential signs of depression.</p>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.image("D:\Project\DP.jpeg", caption="", width=150)

    # Input Section
    user_input = st.text_area("Enter your text here")

    # Processing Section
    if st.button("Detect"):
        if user_input.strip():
            # Preprocess the input
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()  

            # Map prediction to labels
            result = "Depression detected." if prediction == 1 else "Non-depression detected."
            st.success(result)
        else:
            st.error("Please enter some text for analysis.")

elif page == "Settings":

    # Provide a description of the settings
    st.markdown(
        """
        <p style='font-style: italic; font-size: 18px;'>choose the display mode of the app. following:</p>
        """,
        unsafe_allow_html=True,
    )
    # Settings Section: Mode Selection
    mode = st.radio("Select Mode", ["Dark Mode", "Light Mode", "System Mode"])
    
    # Set styles according to the selected mode
    set_styles(mode)

    # Provide a description of the settings
    st.markdown(
        """
        <p style='font-style: italic; font-size: 18px;'>In this section, you can choose the display mode of the app. Select one of the following:</p>
        """,
        unsafe_allow_html=True,
    )
    