**Transparent Diagnosis for Diabetic Retinopathy**


**Overview**

> This project focuses on developing an AI-powered system for early and transparent detection of Diabetic Retinopathy (DR) using retinal fundus images.

> The model is designed to provide both accurate predictions and explainable insights using deep learning and Explainable AI (XAI) techniques such as Grad-CAM.


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



**System Architecture**

> User Uploads Image → Preprocessing & Augmentation → Model Prediction (EfficientNet) → Grad-CAM Visualization → Display Result + Report Generation



**System Features**

> Experience the working prototype of our web-based diagnostic system through the Interactive Frontend Portal.
> The web application provides a seamless interface where users can:

>1.Upload retinal fundus images for diabetic retinopathy detection.

>2.View predictions in real time using the trained EfficientNet model.

>3.Analyze visual explanations generated through Grad-CAM heatmaps for better interpretability.

>4.Download personalized diagnostic reports summarizing the disease stage, cause, and possible treatment suggestions.

>Access the Live Web Application: http://127.0.0.1:5000/



**Testing & Results**

> Achieved 97.4% classification accuracy with the EfficientNet model.

> Evaluated performance using metrics like precision, recall, F1-score, and confusion matrix.
>
>  Validated predictions through Grad-CAM visualizations to ensure model reliability.



**Output**

>Predicted Stage of DR

>Grad-CAM visualization of affected retinal area

>Downloadable Diagnostic Report (Disease, Cause, Treatment)


**Future Enhancements**

>Extend the model to detect other eye disorders (Glaucoma, Macular Degeneration).

>Integrate live retinal image capture from a webcam or fundus camera.

>Cloud deployment for real-time tele-ophthalmology support.


**Acknowledgment**

>We would like to express our heartfelt gratitude to **Deepti N N, Assistant Professor**, our project guide at **Rajiv Gandhi Institute of Technology**, for their constant support, valuable insights, and encouragement throughout the development of this project.
>Their guidance played a crucial role in helping us complete the project successfully and achieve meaningful results.



**Team Role Division**

>Saranya M S – Data, Model Development & Documentation:
Collected and preprocessed the dataset, performed image augmentation, and trained the EfficientNet model for diabetic retinopathy detection, and prepared the final project report and documentation.

>Veena K – Explainable AI & Architecture:
Integrated Grad-CAM for visual interpretability and designed the system architecture, workflow, and analysis diagrams.

>Ruthu N – Web Development & Deployment:
Developed the Flask-based web application, connected frontend (HTML, CSS, JS) with backend, and implemented the diagnostic report download feature.

>Spandana R – Testing:
Conducted model evaluation using accuracy, precision, recall, and F1-score metrics, performed software testing.
