import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# read data
data = pd.read_csv('spam.csv', encoding='latin-1')

# drop unuse column
data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

# features and labels
data['label'] = data['class'].map({'ham':0, 'spam':1})

X = data['message']
y = data['label']

# extract feature with count vectorize
cv = CountVectorizer()
X = cv.fit_transform(X)

pickle.dump(cv, open('tranform.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# naive bayes classifire
clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))


filename = 'nlp_model.pkl'
pickle.dump(clf, open(filename, 'wb'))





