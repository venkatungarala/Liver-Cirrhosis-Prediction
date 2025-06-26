# 🩺 Liver Cirrhosis Prediction Web App

This is a machine learning-powered web application that predicts the risk of liver cirrhosis based on lifestyle and clinical features. 
It is built using Flask for the backend and a Random Forest Classifier trained on real medical data.

---

## Features

1.Real-time liver disease prediction
2.Clean and responsive HTML/CSS UI
3.Input includes medical history & lifestyle habits
4.Powered by a trained Random Forest model


---

## Demo

Check out the live demo here:  
[Your GitHub Pages URL](https://liver-cirrhosis-prediction-1.onrender.com)

---

## 📊 Dataset Used


Source: Indian Liver Patient Dataset (ILPD)

Published By: UCI Machine Learning Repository

Link: https://archive.ics.uci.edu/ml/datasets/ILPD+%28Indian+Liver+Patient+Dataset%29

## Dataset Details:

583 patient records

10 features + 1 target variable (liver disease status)

Preprocessing included handling missing values and scaling for consistency

---
## 🧾 Input Fields Used in the App
The model takes the following user inputs:

Age – Age of the user in years

Gender – Male / Female

Body Mass Index (BMI) – BMI score (e.g. 22.5)

Alcohol Consumption – Units of alcohol per week (e.g. 3.5)

Do you smoke? – Yes / No

Family history of liver disease? – Yes / No

Do you exercise regularly? – Yes / No

Do you have diabetes? – Yes / No

Do you have high blood pressure? – Yes / No

Liver Function Test Result – A liver function score (e.g. 34.7)

---

## 📈 Feature Impact on Prediction
Based on model training and feature importance from the dataset:

Liver Function Test Result: Most influential in predicting disease

Alcohol Consumption: Strong risk factor

BMI: Higher BMI increases risk

Age: Older age increases chances of liver-related issues

Smoking and Family History: Moderate impact

Exercise: Positive protective factor

Diabetes and High Blood Pressure: Contribute to risk but less than direct liver indicators

🧠 Feature importance was extracted using .feature_importances_ from the RandomForestClassifier.

---
## 🧰 Tech Stack
Frontend: HTML5, CSS3

Backend: Python 3.10, Flask 2.3.3

ML Libraries:

scikit-learn 1.3.0

pandas

numpy

joblib

---

## 📁 Project Structure
📦 liver-cirrhosis-prediction/
├── training/               # Notebook for training the model
├── models/                 # Trained model and scaler (.pkl files)
├── templates/              # index.html (user form)
├── app.py                  # Flask backend logic
├── requirements.txt        # Required Python packages
└── Procfile                # For Render deployment

---


## 🖥️ How to Run Locally

Clone the repo

Install dependencies using pip install -r requirements.txt

Run the Flask app using python app.py

---


## 👨‍💻 Author
Venkat Ungarala
B.Tech CSE Student
Passionate about AI and Full-Stack Development
GitHub: https://github.com

---
 ## 📄 License
This project is open source and free to use.

## 💬 Feedback
Found a bug or have suggestions to improve?
Feel free to open an issue or submit a pull request!

## Made with ❤️ using Python, Flask, and Machine Learning



