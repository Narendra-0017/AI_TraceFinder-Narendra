# TraceFinder: AI-Powered Forensic Scanner Identification 🕵️‍♂️

**TraceFinder** is a web application that identifies the source scanner used to create a digital image. By analyzing the unique, invisible artifacts and noise patterns left by a scanner's hardware, our machine learning model can trace an image back to its origin device. This has powerful applications in digital forensics, document authentication, and evidence verification.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai--tracefinder.streamlit.app/)

**Check out the live demo:** **[https://ai--tracefinder.streamlit.app/](https://ai--tracefinder.streamlit.app/)**

![TraceFinder Screenshot](https://raw.githubusercontent.com/your-username/your-repo-name/main/screenshot.png)
*(**Note**: To make the image above work, please add a screenshot of your app to your repository and update this link!)*

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

* **Dataset & Metadata**: [**Link to your `TraceFinder` Google Drive Folder**]([your-link-here](https://drive.google.com/drive/folders/1NLErgKgCGQwES5D8L4OtnF9YdcylRlDZ?usp=sharing))
* **PRNU Wavelets & Trained Models**: [**Link to your `PRNU_Residual_Wavelet` Google Drive Folder**]([your-link-here](https://drive.google.com/drive/folders/1fFXaTnwRhX_y30Vk-Kmd3wZ93E930JSF?usp=sharing))

***

## 📂 Project Structure

The project is organized into several key directories that handle data, features, and models.

### `TraceFinder/`
This directory contains the primary dataset and processed files.
