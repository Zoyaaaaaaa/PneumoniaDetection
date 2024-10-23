import pandas as pd
from PIL import Image
import tensorflow as tf
import calendar 
import google.generativeai as genai
from datetime import datetime
from PIL import Image
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import calendar 
from datetime import datetime, timedelta
import streamlit as st

# Load the pre-trained model
model = tf.keras.models.load_model('pneumonia_detection_model.h5')

# Configure page layout and theme
st.set_page_config(page_title="Lung Health Care Plan Generator", layout="wide", initial_sidebar_state="expanded")

# Custom CSS to improve the app's appearance
st.markdown("""
<style>
    .main {
        background-color: #e6f3ff;
    }
    .stButton>button {
        background-color: #4682b4;
        color: white;
        border-radius: 10px;
        border: 2px solid #4169e1;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4169e1;
        border-color: #1e90ff;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stTextArea>div>div>textarea {
        background-color: #f0f8ff;
        border-radius: 5px;
        border: 1px solid #b0c4de;
    }
    h1, h2, h3 {
        color: #191970;
    }
    .sidebar .sidebar-content {
        background-color: #e6e6fa;
    }
    .stRadio > label {
        background-color: #e6e6fa;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)
genai.configure(api_key="AIzaSyCIb89ZG8R2VfEChi07w9ze2o_yyBYZO_g")

# Function to load Google Gemini API and get response
def get_gemini_response(image, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([image[0], prompt])
    return response.text

def input_image_setup(uploaded_file):
    """Setup the image for API processing."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{
            "mime_type": uploaded_file.type,
            "data": bytes_data
        }]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
def generate_summary(prompt):
    """Generate a summary using Google's Gemini model."""
    try:
        genai.configure(api_key="AIzaSyCIb89ZG8R2VfEChi07w9ze2o_yyBYZO_g")  # Replace with your actual API key
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )
        
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error in generating summary: {str(e)}")
        return None

def collect_patient_info():
    """Collect patient details from user input."""
    with st.form("patient_form"):
        st.subheader("📋 Patient Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name = st.text_input("Patient's Name")
            age = st.number_input("Age", min_value=0, max_value=120)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        with col2:
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, step=0.1)
            height = st.number_input("Height (cm)", min_value=0, max_value=300)
            blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
        
        with col3:
            smoking_status = st.selectbox("Smoking Status", ["Non-smoker", "Former smoker", "Current smoker"])
            occupation = st.text_input("Occupation")
        
        st.subheader("🫁 Lung Health Information")
        col1, col2 = st.columns(2)
        
        with col1:
            cough_type = st.selectbox("Cough Type", ["Dry", "Productive", "Wheezing", "None"])
            cough_duration = st.selectbox("Cough Duration", ["Less than 1 week", "1-2 weeks", "2-4 weeks", "More than 4 weeks"])
            sputum_color = st.selectbox("Sputum Color", ["Clear", "White", "Yellow", "Green", "Blood-tinged", "None"])
        
        with col2:
            breathing_difficulty = st.slider("Breathing Difficulty (0-10)", 0, 10, 5)
            chest_pain = st.selectbox("Chest Pain", ["None", "Mild", "Moderate", "Severe"])
            fever = st.number_input("Current Temperature (°C)", min_value=35.0, max_value=42.0, step=0.1)
        
        symptoms = st.multiselect("Other Symptoms", 
            ["Fatigue", "Loss of appetite", "Night sweats", "Rapid heartbeat", "Confusion", "Blue lips or fingernails"])
        comorbidities = st.multiselect("Comorbidities", 
            ["Asthma", "COPD", "Diabetes", "Heart disease", "Hypertension", "Immunocompromised", "None"])
        allergies = st.text_area("Allergies (if any)")
        current_medications = st.text_area("Current Medications")
        
        submitted = st.form_submit_button("Generate Care Plan")
        
        if submitted:
            return {
                "name": name, "age": age, "gender": gender, "weight": weight,
                "height": height, "blood_type": blood_type, "smoking_status": smoking_status,
                "occupation": occupation, "cough_type": cough_type, "cough_duration": cough_duration,
                "sputum_color": sputum_color, "breathing_difficulty": breathing_difficulty,
                "chest_pain": chest_pain, "fever": fever, "symptoms": ", ".join(symptoms),
                "comorbidities": ", ".join(comorbidities), "allergies": allergies,
                "current_medications": current_medications
            }
    return None

def display_summary(summary):
    """Display the generated care plan summary."""
    if summary:
        st.success("Care Plan Generated Successfully!")
        st.markdown("### 📜 Personalized Lung Health Care Plan")
        st.write(summary)
    else:
        st.error("Unable to generate care plan. Please try again.")

def main():
    st.sidebar.title("📍 Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Pneumonia Detection", "Patient Information", "Care Plan", "Tips and Resources","Post Treatment"])
    
    if page == "Home":
        st.title("PneumoTrack")
        st.header("Pneumonia Detection Through Chest X-ray")
        st.write("Our Pneumonia Detection model leverages deep learning algorithms to analyze chest X-rays and determine the likelihood of pneumonia.")

    elif page == "Pneumonia Detection":
        st.title("Pneumonia Predictor")
        st.markdown("<h4 style='font-size:22px; font-weight:bold; color:lightblue;'>Upload a Chest X-ray Image</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)

            if st.button("Predict"):
                # Preprocess the image and make a prediction
                image = image.convert("L").resize((150, 150))
                image_array = np.array(image) / 255.0
                image_array = np.expand_dims(image_array, axis=-1)
                image_array = np.expand_dims(image_array, axis=0)
                prediction = model.predict(image_array)

                # Display prediction result
                st.subheader("Prediction Result")
                if prediction[0][0] > 0.5:
                    st.warning("The model predicts: Pneumonia detected.")
                else:
                    st.success("The model detects *no signs of pneumonia* in the uploaded image.")

    elif page == "Patient Information":
        st.title("🫁 Pneumonia Care Plan Generator")
        patient_info = collect_patient_info()
        
        if patient_info:
            st.session_state['patient_info'] = patient_info
            st.success("Patient information saved. Please go to the Care Plan tab to generate the plan.")

    elif page == "Care Plan":
        st.header("📝 Generate Care Plan")
        if 'patient_info' in st.session_state:
            patient_info = st.session_state['patient_info']
            prompt = f"""
            Create a comprehensive, personalized lung health care plan for a {patient_info['age']} year-old {patient_info['gender']} patient.
            Patient Details: {patient_info}
            
            Please include:
            1. Detailed treatment recommendations
            2. Medication plan (considering current medications and allergies)
            3. Lifestyle adjustments specific to lung health
            4. Diet recommendations to support respiratory health
            5. Exercise and breathing exercise recommendations
            6. Follow-up schedule with specific tests or check-ups
            7. Warning signs specific to their condition to watch for
            8. Home care instructions for managing symptoms
            9. Recommendations for improving air quality at home and work
            10. Mental health considerations and stress management techniques
            """
            
            with st.spinner("Generating personalized lung health care plan..."):
                care_plan = generate_summary(prompt)
                if care_plan:
                    display_summary(care_plan)
                else:
                    st.error("Unable to generate care plan. Please check your API key and try again.")
        else:
            st.warning("Please fill out the Patient Information first.")

    elif page == "Tips and Resources":
        st.title("Tips and Resources for Lung Health Management")
        
        tabs = st.tabs(["Tips for Managing Lung Health", "Lung Health Resources"])
        
        with tabs[0]:
            st.subheader("🌟 Tips for Managing Lung Health")
            tips = [
                "Practice deep breathing exercises daily",
                "Use a humidifier to keep air moist",
                "Avoid smoking and secondhand smoke",
                "Stay hydrated to help thin mucus",
                "Get vaccinated against pneumonia and flu",
                "Exercise regularly to improve lung capacity",
                "Maintain good indoor air quality",
                "Follow a balanced diet rich in antioxidants"
            ]
            
            for tip in tips:
                st.markdown(f"<p style='font-size: 18px;'>- {tip}</p>", unsafe_allow_html=True)
        
        with tabs[1]:
            st.subheader("📚 Lung Health Resources")
            resources = [
                "[American Lung Association](https://www.lung.org)",
                "[CDC Pneumonia Overview](https://www.cdc.gov/pneumonia/index.html)",
                "[World Health Organization - Respiratory Diseases](https://www.who.int/health-topics/respiratory-diseases)",
                "[European Respiratory Society](https://www.ersnet.org/)",
                "[Lung Foundation Australia](https://lungfoundation.com.au/)"
            ]
            for resource in resources:
                st.markdown(resource)




    elif page == "Post Treatment":
        st.header("🍽️ Calorie Advisor")
        st.write("Upload an image of your meal (breakfast, lunch, or dinner).")

        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        image = ""

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)

            submit = st.button("Analyze calories of my meal")
            input_prompt = """
               You are an expert nutritionist. Analyze the food items in the uploaded image and provide a general calorie breakdown along with suggestions on how these foods can aid recovery for pneumonia patients.

For each food item, list the estimated calorie count in the following format:

1. Item 1 - approximately X calories
2. Item 2 - approximately Y calories
3. Total Calories - Z calories

Additionally, provide brief commentary on how these foods can contribute to overall health and recovery, especially for those recovering from pneumonia. Consider including foods that are high in nutrients beneficial for lung health and immune support.
                """


            # If submit button is clicked
            if submit:
                image_data = input_image_setup(uploaded_file)
                response = get_gemini_response(image_data, input_prompt)
                st.subheader("🍽️ Calorie Breakdown")
                st.write(response)
           
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
