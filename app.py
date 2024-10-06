import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

# Load the classifier and CountVectorizer (assuming they are stored in .sav files)
with open(r"D:\vs code\Restaurant reviews\multinomialNB.sav", 'rb') as model_file:
    classifier = pickle.load(model_file)

with open(r"D:\vs code\Restaurant reviews\count_vectorizer.sav", 'rb') as cv_file:
    cv = pickle.load(cv_file)

# Function to clean and predict sentiment
def predict_sentiment(sample_review):
    sample_review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=sample_review)
    sample_review = sample_review.lower()
    sample_review_words = sample_review.split()
    sample_review_words = [word for word in sample_review_words if word not in set(stopwords.words('english'))]
    
    ps = PorterStemmer()
    final_review = [ps.stem(word) for word in sample_review_words]
    final_review = ' '.join(final_review)

    temp = cv.transform([final_review]).toarray()
    return classifier.predict(temp)

# Streamlit app interface
st.title('Restaurant reviews Analysis App')

# Input field for review text
sample_review = st.text_area("Enter a review:")

if st.button('Predict'):
    prediction = predict_sentiment(sample_review)
    
    if prediction == 1:
        st.success('This is a POSITIVE review!.')
    else:
        st.error('This is a NEGATIVE review!')
