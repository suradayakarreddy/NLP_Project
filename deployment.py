#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import string
from nltk.corpus import stopwords


# In[2]:


model = joblib.load("model.pkl")
# Load the saved CountVectorizer
vectorizer = joblib.load("vectorizer.pkl")


# In[3]:


stop_words = set(stopwords.words("english"))
punctuation = string.punctuation


# In[4]:


def clean_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = "".join([char for char in text if char not in punctuation])
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


# Title of the app
st.title("Fake News Classifier")


# In[5]:


st.write("Enter news text below to classify:")
news_input = st.text_area("")


# In[6]:


if st.button("Classify"):
    if news_input:
        cleaned_text = clean_text(news_input)
        # Preprocess the input using the same vectorizer
        input_vectorized = vectorizer.transform([cleaned_text])
        # Make prediction
        prediction = model.predict(input_vectorized)

        # Display the result
        st.write("The news is", prediction[0])
    else:
        st.write("Please enter some news text.")


# In[ ]:




