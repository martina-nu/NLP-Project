import pandas as pd
import pickle
import numpy as np
import re
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



url = "https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv"

df_raw = pd.read_csv(url)

df = df_raw.copy()
#Remove duplicates
df = df.drop_duplicates().reset_index(drop = True)

#Encode
df['is_spam'] = df['is_spam'].apply(lambda x: 1 if x == True else 0)

def no_alpha(text): #remove non-alphanumeric characters
    return re.sub("(\\d|\\W)+"," ",text)

def tags(text): #remove tags
    return re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

def punct(text):#remove punctuation
    return re.sub('[^a-zA-Z]', ' ', text)

def url(text): #remove start of url
    return re.sub(r'(https://www|https://)', '', text)

df['clean_url'] = df['url'].apply(url).apply(no_alpha).apply(tags).apply(punct)


X = df['clean_url']
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=15, stratify=y)

vec = CountVectorizer(stop_words='english')
X_train = vec.fit_transform(X_train).toarray()
X_test=vec.transform(X_test).toarray()

classifier = SVC(C=10, gamma= 0.1, kernel='rbf')
classifier.fit(X_train, y_train)

filename = 'models/svc_model.sav'
pickle.dump(classifier, open(filename,'wb'))