# 🌿 Leaf Disease Detection

This project focuses on detecting **33 types of leaf diseases** using deep learning and transfer learning techniques. It leverages the power of pre-trained models to classify plant leaf images with high accuracy, providing a powerful tool for modern agriculture and plant health monitoring.

---

## 🚀 Tech Stack & Tools

![Python](https://img.shields.io/badge/-PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/-Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

---

## 📁 Project Structure
```
├── data/ # Contains training and test datasets
├── model/ # Pretrained model files (if any)
├── main.py # Streamlit app for running the model
├── requirements.txt # List of required libraries
└── README.md # Project overview
```
---

## 📦 Installation & Usage

1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```
2.Run the app using Streamlit:
```
streamlit run main.py
```
🧠 Model Details
The leaf disease detection model uses transfer learning with a pre-trained CNN architecture (like VGG16, ResNet, or EfficientNet). The final model was fine-tuned on a dataset of 33 disease categories of plant leaves.

📌 Model goals:

Help farmers and agronomists detect issues early

Support smart agriculture initiatives

📚 Dataset
The dataset contains:

Images of healthy and diseased leaves

33 labeled categories

Preprocessed for training and evaluation

📌 Future Scope
Integration with mobile camera apps

Real-time disease alert systems

Integration with weather data and soil info for precision agriculture

🙋‍♀️ Author
Made with ❤️ by Deepika
Feel free to connect or contribute!

