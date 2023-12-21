from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn. model_selection import train_test_split
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

app = Flask(__name__)

df = pd.read_csv('D:\stress.csv')

nltk. download( 'stopwords' )
stemmer = nltk. SnowballStemmer("english")
stopword=set (stopwords . words ( 'english' ))

def clean(text):
    text = str(text) . lower()  
    text = re. sub('\[.*?\]',' ',text)  
    text = re. sub('https?://\S+/www\. \S+', ' ', text)
    text = re. sub('<. *?>+', ' ', text)
    text = re. sub(' [%s]' % re. escape(string. punctuation), ' ', text)
    text = re. sub(' \n',' ', text)
    text = re. sub(' \w*\d\w*' ,' ', text)
    text = [word for word in text. split(' ') if word not in stopword] 
    text =" ". join(text)
    text = [stemmer . stem(word) for word in text. split(' ') ]
    text = " ". join(text)
    return text
df [ "text"] = df["text"]. apply(clean)

x = np.array (df["text"])
y = np.array (df["label"])
cv = CountVectorizer ()
X = cv. fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.25)

model=BernoulliNB()
model.fit(xtrain,ytrain)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    cleaned_input = clean(user_input)
    data = cv.transform([cleaned_input]).toarray()
    output = model.predict(data)
    return str(output[0])

if __name__ == '__main__':
    app.run(debug=True)
