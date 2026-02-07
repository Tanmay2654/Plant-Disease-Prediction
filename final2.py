import streamlit as st
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index, prediction, input_arr

# Plot class probabilities
def plot_predictions(predictions, class_names):
    fig, ax = plt.subplots(figsize=(6, 8))
    y_pos = np.arange(len(class_names))
    ax.barh(y_pos, predictions[0], align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_title('Model Prediction Confidence')
    st.pyplot(fig)

# Full Disease Information Dictionary (simplified here; replace with full 38-class dictionary)
disease_info_full = {
    "Apple___Apple_scab": {
        "description": "Apple scab is a fungal disease causing dark, scabby lesions on leaves and fruit, which can lead to fruit deformities and premature leaf drop.",
        "prevention": "Plant resistant varieties and ensure proper air circulation by pruning and spacing trees appropriately.",
        "cure": "Apply fungicides early in the season and remove fallen leaves and infected fruit."
    },
    "Tomato___Late_blight": {
        "description": "Brown lesions and white fungal growth on leaf undersides; affects leaves and fruit.",
        "prevention": "Avoid moisture buildup and use resistant varieties.",
        "cure": "Apply metalaxyl-based fungicides and remove infected tissue."
    },
    "Apple___healthy": {
        "description": "Your apple plant appears healthy.",
        "prevention": "Maintain proper watering, pruning, and monitoring.",
        "cure": "No treatment necessary."
    }
    # Add full 38-disease dictionary entries here
}

# Plant Care Tips
tips = {
    "Tomato": "Water at soil level. Avoid overhead watering. Use mulch to retain moisture.",
    "Apple": "Thin fruit for better air flow. Watch for fungal issues in early spring.",
    "Corn": "Avoid dense planting. Monitor for leaf spot and rust diseases.",
    "Potato": "Ensure crop rotation and keep soil moist but not soggy."
}

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("Home image not found: home_page.jpeg")
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Upload a plant image and our system will predict if it has any diseases.

    ### Features:
    - Deep learning-based disease prediction
    - Confidence score & probability chart
    - Description, prevention, and cure info
    - Report download and prediction history
    - General care tips for the plant
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    This project uses a deep learning model trained on the PlantVillage dataset to classify plant diseases. It includes 38 different classes of healthy and diseased crop leaves.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    if "history" not in st.session_state:
        st.session_state.history = []

    test_image = st.file_uploader("Choose an Image:")
    if test_image:
        st.image(test_image, use_container_width=True, caption="Uploaded Image")

    if st.button("Predict"):
        if test_image:
            st.snow()
            st.write("Our Prediction")

            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
                'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
            ]

            result_index, prediction, input_arr = model_prediction(test_image)
            predicted_label = class_name[result_index]
            confidence = prediction[0][result_index]

            st.success(f"🌿 Prediction: **{predicted_label}**")
            st.info(f"🧮 Confidence: {confidence * 100:.2f}%")
            plot_predictions(prediction, class_name)
            st.image(input_arr[0].astype(np.uint8), caption="Model Input View", use_container_width=True)

            info = disease_info_full.get(predicted_label)
            if info:
                st.markdown(f"""
                ### 🦠 About the Disease
                {info['description']}

                ### 🛡️ Prevention
                {info['prevention']}

                ### 💊 Cure / Treatment
                {info['cure']}
                """)

                plant_type = predicted_label.split("___")[0].split("_(")[0]
                if plant_type in tips:
                    st.info(f"🌿 General care tips for **{plant_type}**:\n{tips[plant_type]}")

                report = f"""
                PLANT DISEASE PREDICTION REPORT
                ----------------------------------
                Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                Prediction: {predicted_label}
                Confidence: {confidence * 100:.2f}%

                About the Disease:
                {info['description']}

                Prevention:
                {info['prevention']}

                Cure:
                {info['cure']}
                """
                st.download_button(
                    label="📄 Download Report",
                    data=report,
                    file_name=f"{predicted_label.replace(' ', '_')}_report.txt",
                    mime="text/plain"
                )
            else:
                st.warning("No detailed information available for this disease.")

            st.session_state.history.append((predicted_label, datetime.now().strftime("%H:%M:%S")))
            st.markdown("### 📜 Prediction History")
            for item in st.session_state.history:
                st.write(f"🕒 {item[1]} — {item[0]}")
        else:
            st.warning("Please upload an image before predicting.")
