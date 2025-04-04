import streamlit as st
import requests

FLASK_API_URL = "http://127.0.0.1:5000/predict"

st.title("EuroSAT Image Classifier")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        try:
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(FLASK_API_URL, files=files)

            if response.status_code == 200:
                data = response.json()
                prediction = data.get("prediction", "Unknown")

                st.write(f"**Prediction:** {prediction}")

                if "class_probabilities" in data:
                    st.write("**Class Probabilities:**")
                    st.json(data["class_probabilities"])
                else:
                    st.warning("No class probabilities received from the API.")
            else:
                st.error(f"API Error: {response.json().get('error', 'Unknown error')}")
        
        except requests.exceptions.ConnectionError:
            st.error("API is not running! Please start the Flask server.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
