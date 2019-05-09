#importing the libraries 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

#importing the dataset 
trainset = pd.read_csv('train_2kmZucJ.csv')

#importing the testset 
testset = pd.read_csv('test_oJQbWVk.csv')

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 7920):
    review = re.sub('[^a-zA-Z]', ' ', trainset['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus)
y = trainset.iloc[:, 1].values

#Tfidfvector
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vector = TfidfVectorizer(max_df=0.70, min_df=4)
X_tidf = tf_idf_vector.fit_transform(corpus)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha = 1.6)
classifier.fit(X, y)

#cleaning the testset 
corpus_test = []
for i in range(0, 1953):
    review = re.sub('[^a-zA-Z]', ' ', testset['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus_test.append(review)
    
#creating bag of words for test set
X_test = cv.transform(corpus_test)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#k-fold cross validation 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_tidf, y = y, cv = 10)
accuracies.mean()
accuracies.std()

#Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'alpha': [1.4,1.5,1.55,1.6,1.65,1.7,1.8,2]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X, y)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#finding best parameter for TF-IDF 
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords)),
    ('clf', MultinomialNB())])
parameters = {
    'tfidf__max_df': (0.25, 0.5, 0.75,1),
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'clf__estimator__alpha': (1e-2, 1e-3)
}

grid_search_tune = GridSearchCV(pipeline, parameters, cv=10, n_jobs=-1)
grid_search_tune.fit(X_tidf, y)

print("Best parameters set:")
print(grid_search_tune.best_estimator_.steps)



#convert numpy array in dataframe 
columns = ['label']
df = pd.DataFrame(y_pred,columns=columns)

ids = testset.iloc[:1953,0].values
columns1 = ['id']
df2 = pd.DataFrame(ids,columns=columns1)

#joining the label with id 
result = pd.concat([df2, df], axis=1)

#converting result to csv file
result.to_csv('test_predictions7.csv', encoding='utf-8', index=False)
