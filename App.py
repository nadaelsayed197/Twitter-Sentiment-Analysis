import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# -------------------------------
# Load model and components
# -------------------------------
model = load_model("sentiment_model.h5")

# Load tokenizer and label encoder (saved during training)
tokenizer = joblib.load("tokenizer.pkl")
encoder = joblib.load("encoder.pkl")

# -------------------------------
# Text cleaning function
# -------------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', str(text))  # remove URLs
    text = re.sub(r'\@\w+|\#', '', text)                     # remove mentions and hashtags
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)               # remove special characters
    text = text.lower()                                      # lowercase
    text = ' '.join([w for w in text.split() if w not in stop_words])  # remove stopwords
    return text

max_len = 100  # must match training setting

# -------------------------------
# Streamlit user interface
# -------------------------------
st.title("Twitter Sentiment Analysis")
st.write("Type a sentence or tweet below to analyze its sentiment:")

user_input = st.text_area("Enter your text here:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    seq = tokenizer.texts_to_sequences([cleaned])
    pad = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(pad)
    label = encoder.inverse_transform([np.argmax(pred)])[0]
    
    st.subheader(f"Sentiment: **{label.capitalize()}**")

    if label == "positive":
        st.success("😊 Positive sentiment detected!")
    elif label == "negative":
        st.error("😞 Negative sentiment detected!")
    else:
        st.info("😐 Neutral sentiment detected.")

