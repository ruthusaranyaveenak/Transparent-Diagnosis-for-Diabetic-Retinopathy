**Transparent Diagnosis for Diabetic Retinopathy**


**Overview**

This project focuses on developing an AI-powered system for early and transparent detection of Diabetic Retinopathy (DR) using retinal fundus images.

The model is designed to provide both accurate predictions and explainable insights using deep learning and Explainable AI (XAI) techniques such as Grad-CAM.


**Objectives**

>Detect various stages of Diabetic Retinopathy automatically using EfficientNet.

>Ensure interpretability and transparency with visual explanations.

>Provide real-time analysis through a Flask-based web application.

>Generate downloadable diagnostic reports containing disease prediction, cause, and treatment suggestions.



**Tools & Technologies Used**

>Programming Language: Python 3.8+

>Libraries: TensorFlow, Keras, NumPy, Pandas, OpenCV, Matplotlib

>Explainable AI: Grad-CAM, LIME

>Web Framework: Flask

>Frontend: HTML5, CSS3, JavaScript

>Database: MySQL (for login & record management)

>Environment: Jupyter Notebook / VS Code



**Model & Methodology**

>Dataset: Retinal fundus images from Kaggle’s Diabetic Retinopathy dataset.

>Data Preprocessing: Resizing, grayscale conversion, CLAHE enhancement, Gaussian blur, and Canny edge detection.

>Data Augmentation: Zoom, rotation (±15°), and horizontal flip to balance dataset classes.

>Model Used: EfficientNet-B0, selected for its superior accuracy and computational efficiency.

>Explainability: Grad-CAM heatmaps show regions influencing the model’s prediction.


**System Features**

>Real-time DR detection from uploaded images.

>Interactive web interface for easy use.

>Transparent visual outputs for clinician trust.

>Downloadable PDF diagnostic report.


**System Architecture**

User Uploads Image → Preprocessing & Augmentation → Model Prediction (EfficientNet) → Grad-CAM Visualization → Display Result + Report Generation


Output

>Predicted Stage of DR

>Grad-CAM visualization of affected retinal area

>Downloadable Diagnostic Report (Disease, Cause, Treatment)


Future Enhancements

>Extend the model to detect other eye disorders (Glaucoma, Macular Degeneration).

>Integrate live retinal image capture from a webcam or fundus camera.

>Cloud deployment for real-time tele-ophthalmology support.
