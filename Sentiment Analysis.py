import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

st.title("Restaurant Review's Sentiment Analyser")
st.markdown("*A Machine Learning Web App, Built with Streamlit, Deployed using Heroku*")
st.write("")

#Taking input from user
sample_message = st.text_area('Enter your review here...',height=250)

# Load the Naive bayes model and CountVectorizer object from disk
model = pickle.load(open('Sentiment_Prediction_model.pkl', 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))

#Predictin function
def predict_sentiment(sample_review):
    sample_review = re.sub('[^a-zA-Z]',' ',string = sample_review)
    sample_review = sample_review.lower()
    sample_review = sample_review.split()
    ps = PorterStemmer()
    sample_review = [ps.stem(word) for word in sample_review if word not in set(stopwords.words('english'))]
    sample_review = ' '.join(sample_review)
    temp = cv.transform([sample_review]).toarray()
    return model.predict(temp)[0]

# Submit button
if st.button('Submit'):
    result = predict_sentiment(sample_message)
    if result==0:
        st.write('*Negative Review*')
    else:
        st.write('*Positive Review*')
