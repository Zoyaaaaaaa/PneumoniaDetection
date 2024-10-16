import streamlit as st
import pandas as pd
from PIL import Image
import plotly.express as px  # Import Plotly for visualization
# from fpdf import FPDF


# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Predictor", "Recommendations", "Dr. Suggestions"))

# Home Page
if page == "Home":
    # Top logo and title
    col1, col2 = st.columns([1, 8])
    with col1:
        logo = Image.open('logo.png')  # Ensure you have a 'logo.png' file in the same directory
        st.image(logo, width=80)
    with col2:
        st.title("Medify")

    # Heading for the page
    st.header("Pneumonia Detection Through Chest X-ray")

    # Full-width image
    st.image("image.png", use_column_width=True)  # Ensure you have an 'xray_image.png' file

    # Subtitle
    st.subheader("About the Model")

    # Explanation paragraph
    st.write("""
    Our Pneumonia Detection model leverages deep learning algorithms to analyze chest X-rays 
    and determine the likelihood of pneumonia. It aims to assist medical professionals in making 
    faster and more accurate diagnoses.
    """)

    # Expanders for additional information
    with st.expander("Data"):
        st.write("""
        The dataset used for training the model consists of thousands of labeled chest X-rays, 
        including cases of pneumonia and healthy lungs.
        """)
        
        # Load and display the dummy dataset
        st.write("Here is a preview of the dataset used:")
        # Load the CSV file
        df = pd.read_csv('sample_apriori_data.csv')  # Make sure the file exists in the same directory
        st.dataframe(df.head())  # Display first few rows of the dataset

    with st.expander("Data Visualization"):
        st.write("""
        Visualizations include heatmaps highlighting the regions of interest on the X-rays 
        where the model detects pneumonia.
        """)

        # Sample data with the attributes: Gender, Age Group, Income Level, Spending Category, and Loyalty
        # Ensure that 'sample_apriori_data.csv' contains these columns
        if 'Gender' in df.columns and 'Age Group' in df.columns:
            # Create a line plot using Plotly
            fig = px.line(df, x='Age Group', y='Income Level', color='Gender', 
                          title="Income Level by Age Group and Gender",
                          labels={"Income Level": "Income Level (in $)", "Age Group": "Age Group"})
            st.plotly_chart(fig)

    with st.expander("Data Accuracy"):
        st.write("""
        Our model achieves an accuracy of 90%, with continuous efforts to improve through more 
        diverse training data and tuning of the model parameters.
        """)

# Predictor Page
# Predictor Page
elif page == "Predictor":
    st.title("Pneumonia Predictor")

    # Customize the label text for the image upload widget using st.markdown with CSS
    st.markdown("<h4 style='font-size:22px; font-weight:bold; color:lightblue;'>Upload a Chest X-ray Image</h4>", unsafe_allow_html=True)
    
    # Image upload widget
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    # Predict button
    if uploaded_file is not None:
        if st.button("Predict"):
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)

            # Dummy prediction result text
            st.subheader("Prediction Result")
            st.write("This is a dummy prediction: The model detects **no signs of pneumonia** in the uploaded image.")


# Recommendations Page
elif page == "Recommendations":
    st.title("Recommendations")
    st.write("This page will provide health recommendations based on the model's prediction.")

# Dr. Suggestions Page
elif page == "Dr. Suggestions":
    st.title("Dr. Suggestions")
    st.write("This page will provide suggestions for doctors based on your location and condition.")

























# gen ai
# import streamlit as st
# # Assuming that we are using a `gemini` library similar to the structure you provided

# # Placeholder for the generative model API call (replace with actual Gemini API)
# class GeminiModel:
#     def getGenerativeModel(self, model):
#         # Mock Gemini model initialization
#         return self

#     async def generateContent(self, prompt):
#         # Mock response to simulate AI response (replace with actual API call logic)
#         response = f"1. Apollo Hospital, Mumbai\n2. Fortis Hospital, Mumbai\n3. Breach Candy Hospital\n4. SevenHills Hospital\n5. Lilavati Hospital\n6. Jaslok Hospital\n7. Kokilaben Hospital\n8. Nanavati Hospital\n9. Global Hospital\n10. Bombay Hospital"
#         return MockResult(response)

# # Mock result class for demonstration purposes
# class MockResult:
#     def __init__(self, response):
#         self.response = response

#     async def response(self):
#         return self

#     async def text(self):
#         return self.response


# # Function to get clinic suggestions using a generative AI model (Gemini)
# async def get_clinic_suggestions(location, condition):
#     # Initialize the generative model (Gemini) and specify the model
#     gemini = GeminiModel()
#     model = gemini.getGenerativeModel(model="gemini-pro")

#     # Define the prompt
#     prompt = f"Suggest the top 10 clinics for {condition} treatment in {location}."

#     try:
#         # Call the model to generate content (suggestions)
#         result = await model.generateContent(prompt)
#         text_response = await result.text()
        
#         # Split the response text into individual suggestions (assuming newline-separated suggestions)
#         suggestions = text_response.split('\n')
#         return suggestions

#     except Exception as e:
#         st.error(f"Error generating clinic suggestions: {str(e)}")
#         return []


# # Streamlit-like page to display the form and suggestions
# def dr_suggestions_page():
#     st.title("Dr. Suggestions")
#     st.write("This page provides suggestions for doctors based on your location and condition.")
    
#     # Input fields for location and condition
#     location = st.text_input("Enter your location (City, State):")
#     condition = st.text_input("Enter the condition (e.g., Pneumonia):", "Pneumonia")

#     if st.button("Get Suggestions"):
#         if location and condition:
#             st.write(f"Fetching the top 10 clinics for {condition} in {location}...")

#             # Get clinic suggestions using AI (asynchronously)
#             suggestions = await get_clinic_suggestions(location, condition)

#             if suggestions:
#                 st.write("Here are the top clinics:")
#                 for idx, suggestion in enumerate(suggestions, 1):
#                     st.write(f"{idx}. {suggestion}")
#             else:
#                 st.write("No suggestions found. Please try again.")
#         else:
#             st.error("Please enter both location and condition.")


# # Run the page (this would be inside the main Streamlit app)
# if __name__ == "__main__":
#     dr_suggestions_page()




#webscrapping
# import requests
# from bs4 import BeautifulSoup
# import streamlit as st

# # Function to scrape the webpage
# def scrape_doctors_data(url):
#     # Send an HTTP request to the given URL
#     response = requests.get(url)
    
#     # If the request was successful
#     if response.status_code == 200:
#         soup = BeautifulSoup(response.text, 'html.parser')
        
#         doctors = []

#         # Find all the doctor cards on the page
#         doctor_cards = soup.find_all('div', class_='c-listing__left')  # Correct class from the HTML structure
        
#         for card in doctor_cards[:6]:  # Limit to 6 doctors
#             # Extract image URL
#             image_tag = card.find('img', class_='profile-photo')  # Modify class to match the image class
#             image_url = image_tag['src'] if image_tag else None
            
#             # Extract doctor name
#             name_tag = card.find('h2', class_='doctor-name')  # Modify class to match the name tag
#             name = name_tag.text.strip() if name_tag else "Unknown"
            
#             if image_url:
#                 doctors.append({"name": name, "image_url": image_url})
        
#         return doctors
#     else:
#         st.error(f"Failed to retrieve data. Status code: {response.status_code}")
#         return []

# # Streamlit app to display doctor profiles
# def display_doctors(doctors):
#     st.title("Doctor Profiles")

#     # Display each doctor in a card
#     for doctor in doctors:
#         st.markdown(f"### {doctor['name']}")
#         st.image(doctor['image_url'], width=150)
#         st.write("---")  # Add a separator between cards

# # URL of the webpage to scrape
# webpage_url = 'https://www.practo.com/mumbai/doctor-for-pneumonia-treatment'  # Replace with actual URL

# # Scrape data and display it
# doctors_data = scrape_doctors_data(webpage_url)

# # Display the doctors data in streamlit
# if doctors_data:
#     display_doctors(doctors_data)


