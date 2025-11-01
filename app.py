import streamlit as st
import google.generativeai as genai
import os
import PIL.Image

# Set API Key for Google Gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyDYLtzKh0FPqyb696KWNi8owmo0NysIhyM"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Load the Gemini Model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")


# Function to analyze human attributes
def analyze_human_attributes(image):
    prompt = """
    You are an AI trained to analyze human attributes from images with high accuracy.
    Carefully analyze the given image and return the following structured details:

    - **Gender**
    - **Age Estimate**
    - **Ethnicity**
    - **Mood**
    - **Facial Expression**
    - **Glasses**
    - **Beard**
    - **Hair Color**
    - **Eye Color**
    - **Headwear**
    - **Emotions Detected**
    - **Confidence Level**

    Now, based on the appearance, also guess the most likely social media accounts or handles this person might use. Include:

    - **Likely Instagram Handle** (e.g., @alex_travels)
    - **Likely Facebook Name** (e.g., Alex J. Smith)
    - **Likely LinkedIn Profile** (e.g., alex-j-smith-92)

    Be creative, but realistic. The IDs should look like a real person might use them.
    """
    response = model.generate_content([prompt, image])
    return response.text.strip()


# Streamlit App UI
st.title("Human Attribute & Look-Alike Detection")
st.write("Upload an image to detect human attributes and possible social media IDs using AI.")

# Image Upload
uploaded_image = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    img = PIL.Image.open(uploaded_image)
    
    with st.spinner("Analyzing the image... Please wait..."):
        person_info = analyze_human_attributes(img)

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### AI Prediction Result")
        st.write(person_info)


