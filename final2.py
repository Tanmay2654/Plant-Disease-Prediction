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
        "description": "Fungal disease causing olive-green or black scabs on leaves and fruit.",
        "prevention": "Plant resistant varieties, prune for air circulation, remove fallen leaves.",
        "cure": "Apply fungicides like captan or sulfur in early spring."
    },
    "Apple___Black_rot": {
        "description": "Fungal infection leading to black rot on fruit and leaf spots.",
        "prevention": "Remove mummified fruit, prune infected branches.",
        "cure": "Use copper-based fungicides and improve sanitation."
    },
    "Apple___Cedar_apple_rust": {
        "description": "Rust fungus causing orange spots on leaves and fruit deformities.",
        "prevention": "Remove nearby cedar/juniper trees, plant resistant apple varieties.",
        "cure": "Apply fungicides like myclobutanil during wet weather."
    },
    "Apple___healthy": {
        "description": "No disease detected; the apple plant is healthy.",
        "prevention": "Maintain proper watering, fertilization, and pest monitoring.",
        "cure": "No treatment needed."
    },
    "Blueberry___healthy": {
        "description": "No disease detected; the blueberry plant is healthy.",
        "prevention": "Ensure acidic soil, good drainage, and mulch.",
        "cure": "No treatment needed."
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "description": "White powdery fungus on leaves and fruit, causing distortion.",
        "prevention": "Improve air circulation, avoid overhead watering.",
        "cure": "Apply sulfur or potassium bicarbonate sprays."
    },
    "Cherry_(including_sour)___healthy": {
        "description": "No disease detected; the cherry plant is healthy.",
        "prevention": "Prune regularly, ensure good sunlight.",
        "cure": "No treatment needed."
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "description": "Gray or tan rectangular spots on leaves, leading to reduced yield.",
        "prevention": "Crop rotation, plant resistant hybrids.",
        "cure": "Apply fungicides like azoxystrobin."
    },
    "Corn_(maize)___Common_rust_": {
        "description": "Reddish-brown pustules on leaves, common in cool, humid weather.",
        "prevention": "Plant resistant varieties, avoid dense planting.",
        "cure": "Fungicides like mancozeb if severe."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "description": "Long, elliptical gray-green lesions on leaves.",
        "prevention": "Crop rotation, remove crop residue.",
        "cure": "Apply fungicides like propiconazole."
    },
    "Corn_(maize)___healthy": {
        "description": "No disease detected; the corn plant is healthy.",
        "prevention": "Balanced fertilization, weed control.",
        "cure": "No treatment needed."
    },
    "Grape___Black_rot": {
        "description": "Black spots on leaves and fruit, leading to shriveling.",
        "prevention": "Prune for air flow, remove mummies.",
        "cure": "Fungicides like captan."
    },
    "Grape___Esca_(Black_Measles)": {
        "description": "Internal wood decay with black streaks on leaves.",
        "prevention": "Avoid wounding vines, use healthy stock.",
        "cure": "Prune out infected parts; no full cure."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "description": "Dark spots with yellow halos on leaves.",
        "prevention": "Improve ventilation, remove debris.",
        "cure": "Copper-based fungicides."
    },
    "Grape___healthy": {
        "description": "No disease detected; the grape plant is healthy.",
        "prevention": "Trellis for support, proper pruning.",
        "cure": "No treatment needed."
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "description": "Yellowing leaves, stunted growth, bitter fruit.",
        "prevention": "Control psyllid insects, use disease-free stock.",
        "cure": "No cure; remove infected trees."
    },
    "Peach___Bacterial_spot": {
        "description": "Dark spots on leaves and fruit, leading to cracking.",
        "prevention": "Plant resistant varieties, avoid wet leaves.",
        "cure": "Copper sprays in dormant season."
    },
    "Peach___healthy": {
        "description": "No disease detected; the peach plant is healthy.",
        "prevention": "Thin fruit, prune annually.",
        "cure": "No treatment needed."
    },
    "Pepper,_bell___Bacterial_spot": {
        "description": "Water-soaked spots turning brown on leaves and fruit.",
        "prevention": "Crop rotation, use clean seeds.",
        "cure": "Copper-based bactericides."
    },
    "Pepper,_bell___healthy": {
        "description": "No disease detected; the bell pepper plant is healthy.",
        "prevention": "Stake plants, mulch soil.",
        "cure": "No treatment needed."
    },
    "Potato___Early_blight": {
        "description": "Concentric rings on lower leaves, yellowing.",
        "prevention": "Crop rotation, mulch.",
        "cure": "Fungicides like chlorothalonil."
    },
    "Potato___Late_blight": {
        "description": "Dark lesions on leaves, white mold underneath.",
        "prevention": "Avoid overhead watering, resistant varieties.",
        "cure": "Remove infected parts, apply metalaxyl."
    },
    "Potato___healthy": {
        "description": "No disease detected; the potato plant is healthy.",
        "prevention": "Hill soil, rotate crops.",
        "cure": "No treatment needed."
    },
    "Raspberry___healthy": {
        "description": "No disease detected; the raspberry plant is healthy.",
        "prevention": "Prune canes, ensure good drainage.",
        "cure": "No treatment needed."
    },
    "Soybean___healthy": {
        "description": "No disease detected; the soybean plant is healthy.",
        "prevention": "Rotate with non-legumes, monitor pests.",
        "cure": "No treatment needed."
    },
    "Squash___Powdery_mildew": {
        "description": "White powdery spots on leaves, reducing photosynthesis.",
        "prevention": "Space plants, avoid shade.",
        "cure": "Sulfur or bicarbonate sprays."
    },
    "Strawberry___Leaf_scorch": {
        "description": "Reddish-purple leaf margins, scorching.",
        "prevention": "Improve air circulation, resistant varieties.",
        "cure": "Fungicides like captan."
    },
    "Strawberry___healthy": {
        "description": "No disease detected; the strawberry plant is healthy.",
        "prevention": "Mulch, rotate beds.",
        "cure": "No treatment needed."
    },
    "Tomato___Bacterial_spot": {
        "description": "Small dark spots on leaves and fruit.",
        "prevention": "Use clean seeds, avoid splashing water.",
        "cure": "Copper sprays."
    },
    "Tomato___Early_blight": {
        "description": "Target-like spots on older leaves.",
        "prevention": "Stake plants, mulch.",
        "cure": "Fungicides like mancozeb."
    },
    "Tomato___Late_blight": {
        "description": "Brown lesions and white fungal growth on leaf undersides; affects leaves and fruit.",
        "prevention": "Avoid moisture buildup and use resistant varieties.",
        "cure": "Apply metalaxyl-based fungicides and remove infected tissue."
    },
    "Tomato___Leaf_Mold": {
        "description": "Yellow spots on upper leaves, gray mold below.",
        "prevention": "Improve ventilation, reduce humidity.",
        "cure": "Fungicides like chlorothalonil."
    },
    "Tomato___Septoria_leaf_spot": {
        "description": "Small circular spots with dark borders on leaves.",
        "prevention": "Remove lower leaves, mulch.",
        "cure": "Fungicides like azoxystrobin."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "description": "Yellow stippling on leaves from mite feeding.",
        "prevention": "Monitor for mites, use predatory insects.",
        "cure": "Insecticidal soaps or miticides."
    },
    "Tomato___Target_Spot": {
        "description": "Concentric rings on leaves and fruit.",
        "prevention": "Crop rotation, sanitation.",
        "cure": "Fungicides like copper."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Yellow curled leaves, stunted growth.",
        "prevention": "Control whiteflies, use resistant varieties.",
        "cure": "No cure; remove infected plants."
    },
    "Tomato___Tomato_mosaic_virus": {
        "description": "Mottled leaves, distorted growth.",
        "prevention": "Use virus-free seeds, sanitize tools.",
        "cure": "No cure; destroy infected plants."
    },
    "Tomato___healthy": {
        "description": "No disease detected; the tomato plant is healthy.",
        "prevention": "Stake, prune suckers, consistent watering.",
        "cure": "No treatment needed."
    }
}

# Plant Care Tips
tips = {
    "Apple": "Prune in winter, thin fruit for better size, monitor for pests in spring.",
    "Blueberry": "Maintain acidic soil (pH 4.5-5.5), mulch with pine needles, water regularly.",
    "Cherry": "Protect from birds with netting, prune after harvest, ensure full sun.",
    "Corn": "Plant in blocks for pollination, fertilize with nitrogen, water deeply.",
    "Grape": "Trellis vines, prune heavily in dormant season, harvest when ripe.",
    "Orange": "Fertilize with citrus mix, protect from frost, control psyllids.",
    "Peach": "Thin fruit early, spray for peach leaf curl, harvest when soft.",
    "Pepper": "Stake tall plants, pick regularly to encourage more fruit, avoid cold.",
    "Potato": "Hill soil around stems, harvest when tops die back, store in cool dark.",
    "Raspberry": "Prune old canes after fruiting, mulch to retain moisture.",
    "Soybean": "Inoculate seeds for nitrogen fixation, harvest when pods are full.",
    "Squash": "Plant in hills, control squash bugs, harvest young for best taste.",
    "Strawberry": "Renew beds every 3 years, mulch with straw, pick ripe berries often.",
    "Tomato": "Water at soil level. Avoid overhead watering. Use mulch to retain moisture."
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
