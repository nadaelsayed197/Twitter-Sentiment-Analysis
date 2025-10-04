# 🧠 Twitter Sentiment Analysis using LSTM

This project performs **sentiment analysis** on tweets using an **LSTM (Long Short-Term Memory)** deep learning model.
The model classifies tweets into **Positive**, **Negative**, or **Neutral** sentiments.

---

## 🚀 Features

* Preprocessing of tweets (removing stopwords, URLs, mentions, etc.)
* Trained **LSTM model** for sentiment classification
* Interactive **Streamlit web app** for real-time predictions
* Saved model, tokenizer, and encoder for reuse

---

## 📂 Project Structure

```
Twitter-Sentiment-Analysis/
│
├── notebooks/
│   ├── Twitter_Sentiment_Analysis.ipynb   # Model training
│   ├── Twitter_Sentiment_App.ipynb        # Deployment setup
│
├── app.py                                 # Streamlit application
├── sentiment_model.h5                     # Trained LSTM model
├── tokenizer.pkl                          # Tokenizer
├── encoder.pkl                            # Label encoder
│
├── requirements.txt                       # Dependencies
├── README.md                              # Project description
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Run locally (VS Code or terminal):

```bash
streamlit run app.py
```

This will start the app at:
👉 `http://localhost:8501`

### 2. Run on Google Colab:

Upload the project files, then use:

```python
!streamlit run app.py & npx localtunnel --port 8501
```

It will generate a temporary public link.

### 3. Deploy on Streamlit Cloud:

* Push the project to GitHub.
* Go to [Streamlit Cloud](https://streamlit.io/cloud).
* Connect your repo → Deploy → Get a permanent link.

---

## 📊 Example

Input:

```
I love this product! It's amazing.
```

Output:

```
Sentiment: Positive 😊
```

---

## 📦 Requirements

* `streamlit`
* `tensorflow`
* `nltk`
* `numpy`
* `joblib`
* `scikit-learn`

---

## ✨ Future Work

* Improve accuracy with **Bidirectional LSTM / GRU**
* Add support for multiple languages
* Visualize training metrics

---

## 👩‍💻 Author

Developed by **[Nada Elsayed]**.
