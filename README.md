# TraceFinder: AI-Powered Forensic Scanner Identification 🕵️‍♂️

[cite_start]**TraceFinder** is a web application that identifies the source scanner used to create a digital image[cite: 3]. [cite_start]By analyzing the unique, invisible artifacts and noise patterns left by a scanner's hardware, our machine learning model can trace an image back to its origin device[cite: 3, 4]. [cite_start]This has powerful applications in digital forensics, document authentication, and evidence verification[cite: 5].

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai--tracefinder.streamlit.app/)

**Check out the live demo:** **[https://ai--tracefinder.streamlit.app/](https://ai--tracefinder.streamlit.app/)**

![TraceFinder Screenshot](https://raw.githubusercontent.com/your-username/your-repo-name/main/screenshot.png)
*(Note: To make the image above work, please take a screenshot of your app, name it `screenshot.png`, and place it in your GitHub repository. Update the link if needed.)*

***

## 🚀 Key Features

* [cite_start]**Scanner Identification**: Upload a scanned image and instantly get a prediction of the probable source scanner model[cite: 54].
* [cite_start]**Confidence Score**: Each prediction is accompanied by a confidence score to indicate the model's certainty[cite: 55, 93].
* [cite_start]**Forensic Analysis**: Based on the principle that every scanner leaves a unique "fingerprint" like noise or texture traces[cite: 4].
* [cite_start]**Explainable AI (XAI)**: Visualizes which parts of the image were most influential in the model's decision using techniques like Grad-CAM[cite: 88, 120].

***

## ⚙️ How It Works

[cite_start]The application follows a sophisticated machine learning pipeline to ensure accurate identification, as shown in the system architecture[cite: 24].

![System Architecture](https://raw.githubusercontent.com/your-username/your-repo-name/main/architecture.png)
*(Note: Add the architecture diagram from your PDF to your repository as `architecture.png` and update the link if needed.)*

1.  [cite_start]**Image Input**: A user uploads a scanned image through the web interface[cite: 27, 54].
2.  [cite_start]**Preprocessing**: The image is standardized for the model by resizing, denoising, converting to grayscale, and normalizing pixel values[cite: 43, 44, 64, 65, 66, 67].
3.  [cite_start]**Feature Extraction**: The system extracts scanner-specific features that are often invisible to the naked eye[cite: 18]. These can include:
    * [cite_start]**Noise Patterns**: Using filters like Wavelet or Fast Fourier Transform (FFT)[cite: 46, 73].
    * [cite_start]**Texture Descriptors**: Analyzing patterns using methods like LBP[cite: 47, 74].
4.  [cite_start]**ML/DL Classification**: A pre-trained classifier, such as a **Convolutional Neural Network (CNN)** or **Random Forest**, analyzes the features to match them with a known scanner profile[cite: 19, 49, 50, 51].
5.  [cite_start]**Prediction Output**: The model returns the most likely source scanner model and a confidence score, which are then displayed to the user[cite: 54, 55].

***

## 🎯 Use Cases

[cite_start]This technology is critical in fields where document integrity is paramount[cite: 5].

* [cite_start]**Digital Forensics**: Helps investigators determine which scanner was used to forge or duplicate legal documents[cite: 8]. [cite_start]For example, detecting if a fake certificate was made using a specific scanner[cite: 9].
* [cite_start]**Document Authentication**: Verifies the source of printed and scanned images to detect tampering or fraudulent claims[cite: 10]. [cite_start]This can help differentiate between scans from authorized and unauthorized sources[cite: 11].
* [cite_start]**Legal Evidence Verification**: Ensures that scanned copies submitted in court, like agreements or contracts, originated from known and approved devices[cite: 13, 14].

***

## 🛠️ Tech Stack

* **Backend & ML**: Python
* [cite_start]**Web Framework**: Streamlit [cite: 91]
* [cite_start]**Machine Learning**: Scikit-learn (SVM, Random Forest) [cite: 51][cite_start], TensorFlow/Keras (CNN) [cite: 50]
* **Image Processing**: OpenCV, Pillow
* **Data Handling**: NumPy, Pandas
* [cite_start]**Explainability**: SHAP, Grad-CAM [cite: 88, 120]

***

## 📦 Getting Started Locally

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/AI-TraceFinder.git](https://github.com/your-username/AI-TraceFinder.git)
    cd AI-TraceFinder
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
    *(Note: You need to create a `requirements.txt` file first by running `pip freeze > requirements.txt` in your project's terminal).*
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
Note: Do NOT push large model files (e.g. `haar_hybridcnn.pth`) to GitHub. Use Git LFS or a cloud storage link instead.
