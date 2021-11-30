#The libraties
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from string import punctuation
import bs4 
from bs4 import BeautifulSoup
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#The data input
true_news = pd.read_csv('True.csv')
fake_news = pd.read_csv('Fake.csv')

#Put the data in categories
true_news['category'] = 1
fake_news['category'] = 0

#Merge them into one big dataframe
df = pd.concat([true_news, fake_news])

#Visualise the data by subjects
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.countplot(x='subject', data=df)

#Creat the Corpus
df['text'] = df['title'] + " " + df['text']

#Now I will delete all the columns I don't need
del df['title']
del df['subject']
del df['date']

#Now the largest part (Cleaning the Corpus)
stop = set(stopwords.words('english'))
puncts = list(string.punctuation)
stop.update(punctuation)

def strip_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()
#Removing the square brackets
def remove_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing URLs
def remove_square_brackets(text):
    return re.sub(r'http\s+', '', text)

#Removing the stopwords
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            final_text.append(i.strip())
    return ' '.join(final_text)

#Removing all noisy text
def remove_noisy_text(text):
    text = strip_html(text)
    text = remove_square_brackets(text)
    text = remove_stopwords(text)
    return text

#Then apply the last function to the dataset
df['text'] = df['text'].apply(remove_noisy_text)

#The train test split
x = df.text
y = df.category
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#Creat the count vectorized object
cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1,3))
cv_train = cv.fit_transform(x_train)
cv_test = cv.transform(x_test)

#Creat the TF-IDF object
tfidf = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1,3))
tfidf_train = tfidf.fit_transform(x_train)
tfiddf_test = tfidf.transform(x_test)

#Creating the model with CV
mnb = MultinomialNB()
mnb_cv = mnb.fit(cv_train, y_train)
mnb_cv_predict = mnb_cv.predict(cv_test)

#Creating the model with TF-IDF
mnb_tfidf = mnb.fit(cv_train, y_train)
mnb_tfidf_predict = mnb_tfidf.predict(cv_test)

#The confusion matrix
cm = confusion_matrix(mnb_cv_predict, y_test)