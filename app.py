import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load the model and vectorizer
try:
    model = joblib.load('xgb_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")

def setup_nltk_data():
    try:
        nltk.data.find('corpora/stopwords.zip')
    except LookupError:
        print("Stopwords not found, downloading...")
        nltk.download('stopwords')

setup_nltk_data()

def basic_preprocess(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Fixed regex
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    cleaned_text = ' '.join(filtered_words)
    return cleaned_text

st.title('Fake News Detection')

# Input text
input_text = st.text_area("Enter the news text here:")

if st.button('Predict'):
    if input_text:
        # Preprocess input text
        processed_text = basic_preprocess(input_text)
        
        # Transform text to feature vector
        try:
            vectorized_text = vectorizer.transform([processed_text])
        except Exception as e:
            st.error(f"Error transforming text: {e}")
        
        # Predict
        try:
            prediction = model.predict(vectorized_text)
            if prediction[0] == 0:
                st.write("The news is real.")
            else:
                st.write("The news is fake.")
        except Exception as e:
            st.error(f"Error predicting text: {e}")
    else:
        st.write("Please enter some text to classify.")
