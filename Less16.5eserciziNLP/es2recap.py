import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 

import warnings
warnings.filterwarnings('ignore')

PATH = 'https://frenzy86.s3.eu-west-2.amazonaws.com/python/text_2_clean.csv'

PATTERNS = {
            r"b'": '',
            r'\d+': '',      # rimuove digits (numeri)
            r'[^\w\s]': '',  # Remove punteggiatura e simboli ...,'@!£$%
            r'\b\w{1,2}\b':'',#remove all token less than2 characters
            r'(http|www)[^\s]+':'', # remove website
            r'\s+': ' '      # rimuove tutti i multipli spazi con uno spazio
            }

def clean_column(df, column, patterns):
    for pattern, replacement in patterns.items():
        df[column] = df[column].str.replace(pattern, replacement)
        df[column] = df[column].str.lower() # applica il lower
        df[column] = df[column].str.strip()
    return df

def main():

    df = pd.read_csv(PATH)
    df_cleaned = clean_column(df, 'text', PATTERNS)
    df_cleaned['text'] = df_cleaned['text'].fillna(' ')

    vectorizer = CountVectorizer(max_features=2000, min_df=0.01, max_df=0.9)
    tfidfconverter = TfidfTransformer()

    X_train, X_test, y_train, y_test = train_test_split(df.text, df.sentiment, 
                                                    test_size=0.25, 
                                                    random_state=667
                                                    )
    
    X_train_vect = vectorizer.fit_transform(X_train).toarray()
    X_train = tfidfconverter.fit_transform(X_train_vect).toarray()         
    print(X_train.shape, y_train.shape)

    X_test_vect = vectorizer.transform(X_test).toarray()
    X_test = tfidfconverter.transform(X_test_vect).toarray()
    print(X_test.shape, y_test.shape)

    classifier = SGDClassifier(loss='log', penalty='l2',alpha=1e-3, random_state=667, max_iter=5, tol=None)
    classifier.fit(X_train, y_train) 

    y_pred = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(y_test,y_pred)
    print(f'Accuracy score of the test data : {test_data_accuracy}\n')
    print(classification_report(y_test,y_pred))
    print(f1_score(y_test, y_pred, average='macro'))

    cm = confusion_matrix(y_test, y_pred, labels=classifier.classes_)
    plt.figure(figsize = (10,8))
    sns.heatmap(cm,cmap= "Blues", 
                linecolor = 'black', 
                linewidth = 1, 
                annot = True, 
                fmt='', 
                xticklabels = classifier.classes_, 
                yticklabels = classifier.classes_)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

if __name__ == '__main__':
    main()