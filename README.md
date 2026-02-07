# 🌿 Plant Disease Prediction System

A **web-based application** that uses deep learning to detect and classify plant diseases from leaf images. Built with **Streamlit**, **TensorFlow/Keras**, and trained on the **PlantVillage** dataset (38 classes).

![Home Page Screenshot](home_page.jpeg)
*(Replace with actual screenshot path or upload a good screenshot to the repo)*

## ✨ Features

- Upload a plant leaf image → instant disease prediction
- Shows **predicted disease name** + **confidence score**
- Displays **probability distribution** bar chart
- Detailed information: **description**, **prevention**, **treatment**
- General **plant care tips**
- Download prediction report as text file
- Simple and clean **Streamlit** user interface
- Trained on **Convolutional Neural Network (CNN)**

## Demo Screenshots

| Home Page                          | Prediction Result                              | Probability Chart & Disease Info              |
|------------------------------------|------------------------------------------------|-----------------------------------------------|
| ![Home Page](screenshots/home-page.png) | ![Prediction Result](screenshots/prediction-result.png) | ![Chart & Info](screenshots/chart-and-info.png) |

## 🛠️ Tech Stack

| Category            | Technology / Library               |
|---------------------|-------------------------------------|
| Frontend            | Streamlit                           |
| Backend / ML        | TensorFlow 2.x, Keras               |
| Data Processing     | NumPy, Matplotlib, Pillow           |
| Dataset             | PlantVillage (38 classes)           |
| Model Format        | .keras                              |
| Development         | Python 3.10+                        |

## 📂 Project Structure
Plant-Disease-Prediction/
├── final2.py                       # Main Streamlit application
├── trained_plant_disease_model.keras  # Trained model file
├── requirement.txt                 # Dependencies list
├── Train_plant_disease.ipynb       # Model training notebook
├── Test_plant_disease.ipynb        # Model testing notebook
├── home_page.jpeg                  # Home page image
├── home_page2.jpeg
├── crop diseases prediction synopsis-project.docx   # Project documentation
├── .gitignore
└── README.md
text> **Note:** Large folders (`train/`, `valid/`, `test/`) are not included in the repository due to size.  
> You can download the full PlantVillage dataset from Kaggle.

## 🚀 How to Run Locally

### Prerequisites

- Python 3.10 or 3.11 recommended
- Git (optional)

### Step-by-step

1. **Clone the repository**

```bash
git clone https://github.com/Tanmay2654/Plant-Disease-Prediction.git
cd Plant-Disease-Prediction

Create virtual environment (recommended)

Bash# Windows
python -m venv tf-env
tf-env\Scripts\activate

# macOS / Linux
python3 -m venv tf-env
source tf-env/bin/activate

Install dependencies

Bashpip install -r requirement.txt
(Note: if you renamed the file to requirements.txt, use that name instead)

Run the application

Bashstreamlit run final2.py

Open your browser at:
http://localhost:8501

📊 Model Performance

Trained on PlantVillage dataset (~54k training images)
38 classes (healthy + 37 disease types)
Input image size: 128 × 128
Architecture: Convolutional Neural Network (custom / transfer learning possible)

(You can add accuracy, loss curves, confusion matrix screenshots from your training notebook here)
📄 Project Documentation
Detailed project report:
→ crop diseases prediction synopsis-project.docx
⚡ Future Improvements

Add support for more plant types
Deploy on cloud (Streamlit Community Cloud / Hugging Face Spaces)
Mobile-friendly layout
Real-time camera input
Multi-image / batch prediction
Model explainability (Grad-CAM heatmaps)

❤️ Acknowledgments

Dataset: PlantVillage Dataset on Kaggle
UI Framework: Streamlit
Deep Learning: TensorFlow

📧 Contact / Feedback
Feel free to open an issue or connect:

GitHub: @Tanmay2654


Made with ❤️ for plant health & better farming
Happy predicting! 🌱🔍
