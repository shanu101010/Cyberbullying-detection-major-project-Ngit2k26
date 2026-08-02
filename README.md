# 🛡️ Cyberbullying Detection System

An AI-powered web application that detects cyberbullying in text using Natural Language Processing (NLP) and Deep Learning. The system analyzes user input, predicts whether the text contains cyberbullying, and provides a safe, interactive interface for real-time detection.

---

## 📌 Project Overview

Cyberbullying has become a major concern across social media platforms and online communities. This project aims to assist in identifying harmful or abusive text using machine learning techniques, enabling safer digital communication.

The application combines a trained deep learning model with a Flask web interface to classify user-entered text into cyberbullying or non-cyberbullying categories.

---

## ✨ Features

- 🔍 Real-time cyberbullying detection
- 🤖 Deep Learning based text classification
- 🌐 User-friendly Flask web interface
- 💾 Pre-trained model support
- 📊 User feedback storage
- 📝 Custom abusive word dictionary
- ⚡ Fast prediction for user input

---

## 🛠️ Tech Stack

### Languages
- Python
- HTML
- CSS
- JavaScript

### Frameworks & Libraries
- Flask
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Pickle

---

## 📂 Project Structure

```
Cyberbullying-detection-major-project-Ngit2k26/
│
├── templates/
│   └── index.html
│
├── app.py
├── sentiment_engine.py
├── requirements.txt
├── tokenizer.pkl
├── cyberbullying_model.h5
├── cyberbullying_tweets.csv
├── custom_words.json
├── user_feedback.json
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/shanu101010/Cyberbullying-detection-major-project-Ngit2k26.git
```

### 2. Navigate to the Project

```bash
cd Cyberbullying-detection-major-project-Ngit2k26
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 📊 Dataset

The project uses a labeled cyberbullying dataset for training and evaluation.

Dataset included:

- `cyberbullying_tweets.csv`

---

## 🧠 Model

This repository includes:

- Trained Keras model (`cyberbullying_model.h5`)
- Tokenizer (`tokenizer.pkl`)

> **Note:** The original Hugging Face `model.safetensors` file has been excluded because it exceeds GitHub's 100 MB file size limit.

---


---

## 📈 Future Enhancements

- Multi-class cyberbullying classification
- Toxicity severity score
- Explainable AI (XAI) for predictions
- User authentication
- Deployment using Render or Railway
- REST API support
- Dashboard with prediction analytics

---

## 👩‍💻 Author

**Shanmukhi**

- AIML Undergraduate
- Aspiring Business Analyst & UI/UX Designer

GitHub:
https://github.com/shanu101010

---

## 📄 License

This project is intended for educational and academic purposes.
