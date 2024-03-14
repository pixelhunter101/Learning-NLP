import pandas as pd

messages = pd.read_csv("SMSSpamCollection", sep='\t', names=["label","message"])

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
lem = WordNetLemmatizer()
corpus = []

for i in range(len(messages)):
    review = re.sub('^[a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model_nb = MultinomialNB().fit(X_train, y_train)
y_predict_nb = spam_detect_model_nb.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
spam_detect_model_rf = RandomForestClassifier().fit(X_train, y_train)
spam_detect_model_rf.fit(X_train, y_train)
y_pred_rf = spam_detect_model_rf.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_mat_nb = confusion_matrix(y_test, y_predict_nb)
confusion_mat_rf = confusion_matrix(y_test, y_pred_rf)

from sklearn.metrics import accuracy_score
accuracy_nb = accuracy_score(y_test, y_predict_nb)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
