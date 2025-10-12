# TraceFinder: AI-Powered Forensic Scanner Identification 🕵️‍♂️

**TraceFinder** is a web application that identifies the source scanner used to create a digital image. By analyzing the unique, invisible artifacts and noise patterns left by a scanner's hardware, our machine learning model can trace an image back to its origin device. This has powerful applications in digital forensics, document authentication, and evidence verification.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai--tracefinder.streamlit.app/)

**Check out the live demo:** **[https://ai--tracefinder.streamlit.app/](https://ai--tracefinder.streamlit.app/)**
***

## 🚀 Key Features

* **Scanner Identification**: Upload a scanned image and instantly get a prediction of the probable source scanner model.
* **Confidence Score**: Each prediction is accompanied by a confidence score to indicate the model's certainty.
* **Forensic Analysis**: Based on the principle that every scanner leaves a unique "fingerprint" like noise or texture traces.
* **Explainable AI (XAI)**: Visualizes which parts of the image were most influential in the model's decision using techniques like Grad-CAM.

***

## ⚙️ How It Works

The application follows a sophisticated machine learning pipeline to ensure accurate identification.

1.  **Image Input**: A user uploads a scanned image through the web interface.
2.  **Preprocessing**: The image is standardized by resizing, denoising, converting to grayscale, and normalizing pixel values.
3.  **Feature Extraction**: The system extracts scanner-specific features, including Photo Response Non-Uniformity (PRNU), wavelet transforms (`haar`, `sym2`), and other texture descriptors.
4.  **ML/DL Classification**: A pre-trained classifier (CNN, SVM, or Random Forest) analyzes the features to match them with a known scanner profile.
5.  **Prediction Output**: The model returns the most likely source scanner model and a confidence score.

***

## 💿 Data & Models

The core assets for this project, including the dataset, extracted features, and trained models, can be accessed below.

* [**Dataset & Metadata**: ](https://drive.google.com/drive/folders/1NLErgKgCGQwES5D8L4OtnF9YdcylRlDZ?usp=sharing)
* [**PRNU Wavelets & Trained Models**:](https://drive.google.com/drive/folders/1fFXaTnwRhX_y30Vk-Kmd3wZ93E930JSF?usp=sharing)

***

## 📂 Project Structure

The project is organized into several key directories that handle data, features, and models.

### `TraceFinder/`
This directory contains the primary dataset and processed files.
├── checkpoints/         # Saved model weights from training sessions
├── features/            # Stored feature vectors extracted from images
├── processed_png/       # Preprocessed images ready for analysis
└── metadata.csv         # Maps image files to their correct scanner labels

### `PRNU_Residual_Wavelet.../`
This directory contains the feature engineering artifacts and final trained models.
├── db1/                 # Raw or intermediate dataset source 1
├── db2/                 # Raw or intermediate dataset source 2
├── haar/                # Wavelet features using Haar transform
├── sym2/                # Wavelet features using Symlet 2 transform
└── trained_models/      # The final, serialized machine learning models

***

## 🎯 Use Cases

* **Digital Forensics**: Helps investigators determine which scanner was used to forge or duplicate legal documents.
* **Document Authentication**: Verifies the source of scanned images to detect tampering or fraudulent claims.
* **Legal Evidence Verification**: Ensures that scanned copies submitted in court originated from known and approved devices.

***

## 🛠️ Tech Stack

* **Backend & ML**: Python
* **Web Framework**: Streamlit
* **Machine Learning**: Scikit-learn (SVM, Random Forest), TensorFlow/Keras (CNN)
* **Image Processing**: OpenCV, Pillow, PyWavelets
* **Data Handling**: NumPy, Pandas
* **Explainability**: SHAP, Grad-CAM

***

## 📦 Getting Started Locally

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    *(Create a `requirements.txt` file by running `pip freeze > requirements.txt` in your project's terminal).*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    The application should now be running in your web browser!

***

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
