import streamlit as st
import pandas as pd
import numpy as np
import pickle
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

st.title("Restaurant Review's Sentiment Analyser")
st.markdown("*A Machine Learning Web App, Built with Streamlit, Deployed using Heroku*")
