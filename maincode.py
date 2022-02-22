## Sentimenal Analysis Project


## import essential library 
import numpy as np
import pandas as pd
import warnings
 
 ## load our data
df=pd.read_csv("/content/IMDB Dataset.csv")
df.head()
# get our shape data
df.shape
df.describe()
df['review'].value_counts().sum()
df['sentiment'].value_counts()
df.isnull().sum()
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from wordcloud import WordCloud,STOPWORDS
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from bs4 import BeautifulSoup
import spacy
nltk.download('stopwords')
token=ToktokTokenizer()
stopwords=nltk.corpus.stopwords.words('english')
import re
def noiseremove_text(text):
  soup=BeautifulSoup(text,'html.parser')
  text=soup.get_text()
  text=re.sub('\[[^]]*]& !, @ ,$,%,#,a-zA-z0-9\s,',' ',text)
  return text
  df['review']=df['review'].apply(noiseremove_text)
  df.head()
  def steming(text):
  porstem=nltk.porter.PorterStemmer()
  text=' '.join(([porstem.stem(word) for word in text.split()]))
  return text
  df['review']=df['review'].apply(steming)
  df.head()
from nltk.corpus import stopwords 
from textblob import TextBlob
from textblob import Word
stop_wr=set(stopwords.words('english'))
print(stop_wr)
def remove_stopwords(text,is_lower_case=False):
    tokenizer=ToktokTokenizer()
    tokens=tokenizer.tokenize(text)
    tokens=[i.strip() for i in tokens ]
    if is_lower_case:
      ft=[i for i in tokens if token not in stop_wr]
    else:
      ft=[i for i in tokens if i.lower() not in stop_wr]
      ftexts=' '.join(ft)
    return ftexts
    
    
    df['review']=df['review'].apply(remove_stopwords)
    df.head()
    
    
train_reviews_data=df.review[:3000]
train_sentiments=df.sentiment[:3000]
test_reviews_data=df.review[3000:]
test_sentiments=df.sentiment[3000:
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
cv_train=cv.fit_transform(train_reviews_data)
#transformed tain est reviews
cv_test=cv.transform(test_reviews_data)
print("Bag of Words Train Value:",cv_train.shape)
print("Bag of Words Test Value :",cv_test.shape)
tf=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
#transformed train reviews
tf_train=tf.fit_transform(train_reviews_data)
#transformed test reviews
tf_test=tf.transform(test_reviews_data)
print('Tfidf_train:',tf_train.shape)
print('Tfidf_test:',tf_test.shape)
#labeling the sentient data
label=LabelBinarizer()
#transformed sentiment data
sentiment_data=label.fit_transform(df['sentiment'])
print(sentiment_data.shape)
train_data=df.sentiment[:3000]
train_data.head()
test_data=df.sentiment[3000:]
test_data.head()
from sklearn.linear_model import LogisticRegression
logistic=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
lr_bow=logistic.fit(cv_train,train_data)
print(lr_bow)
lr_tfidf=logistic.fit(tf_train,train_data)
print(lr_tfidf)
lr_bow_predict=logistic.predict(cv_test)
print(lr_bow_predict)
lr_tfidf_predict=logistic.predict(tf_test)
print(lr_tfidf_predict)

from sklearn.metrics import accuracy_score
lr_bow_score=accuracy_score(test_data,lr_bow_predict)
print("lr_bow_score :",lr_bow_score)
lr_tfidf_score=accuracy_score(test_data,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
lr_bow_score=accuracy_score(test_sentiments,lr_bow_predict)
print("lr_bow_score :",lr_bow_score)
#Accuracy score for tfidf features
lr_tfidf_score=accuracy_score(test_sentiments,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)
plt.figure(figsize=(10,10))
positive_text=train_reviews_data[0]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
positive_words=WC.generate(positive_text)
plt.imshow(positive_words,interpolation='bilinear')
plt.show
plt.figure(figsize=(10,10))
positive_text=train_reviews_data[20]
WC=WordCloud(width=1000,height=500,max_words=500,min_font_size=5)
positive_words=WC.generate(positive_text)
plt.imshow(positive_words,interpolation='bilinear')
plt.show
