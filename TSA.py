# -*- coding: utf-8 -*-
"""
TWITTER SENTIMENT ANALYSIS

@authors: Carlos Velázquez y Víctor López

"""

import random
import pandas as pd
import math
import re
import numpy as np
import time
from english_words import get_english_words_set
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn import svm

class Tokenizer:
    
    def __init__(self):
        self.regex_strings = (
        # Phone numbers:
        r"""(?:(?:\+?[01][\-\s.]*)?(?:[\(]?\d{3}[\-\s.\)]*)?\d{3}[\-\s.]*\d{4})""",
        
        # HTML tags:
         r"""<[^>]+>""",
         
        # Twitter username:
        r"""(?:@[\w_]+)""",
        
        # Twitter hashtags:
        r"""(?:\#+[\w_]+[\w\'_\-]*[\w_]+)""",
        
        # URL
        r"""http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+""",
        
        # Remaining word types:
        r"""(?:[a-z][a-z'\-_]+[a-z])|(?:[+\-]?\d+[,/.:-]\d+[+\-]?)|(?:[\w_]+)|(?:\.(?:\s*\.){1,})|(?:\S)""")
            
        self.word_re = re.compile(r"""(%s)""" % "|".join(self.regex_strings), re.VERBOSE | re.I | re.UNICODE)
    
        self.dictionary = get_english_words_set(['web2'], lower=True)
        
    
    def tokenize(self, text):
        
        # Lowercase text
        text = text.lower()
        
        # Repeated Punctuation sign normalization
        text = re.sub(r'(\!{2,})', r' exclamations ', text)
        text = re.sub(r'(\?{2,})', r' questionMarks ', text)
        text = re.sub(r'(\.{3,})', r' ellipsis ', text)

        
        # Substitution and normalization
        tokens = self.word_re.findall(text)
        sub_tokens = []
        for token in tokens:
            
            # Users
            if token.startswith('@'):
                new_token = "user"
                
            # Hashtag
            elif token.startswith('#'):
                new_token = token[1:]
                
            # URLs
            elif token.startswith('http'):
                new_token = "url"
            
            # Normalization
            else:
                new_token = self.normalize(token)
                
            sub_tokens.append(new_token)
            
            
        # Negation
        neg = False
        neg_tokens = []
        punctuation = ['.','?','!',';', 'exclamations', 'questionMarks', 'ellipsis']
        for token in sub_tokens:
            
            if token in punctuation:
                neg = False
            
            if neg:
                neg_token = "NOT_" + token
                
            else:
                neg_token = token
            
            if '\'t' in token or token == 'not':
                neg = True

            neg_tokens.append(neg_token)
        
        return neg_tokens
    
    
    def normalize(self, token):

        new_token = token
        while new_token not in self.dictionary:
            
            # Go through string and search repeated letters
            currentLetter = token[0]
            currentConsecutive = 1
            maxConsecutive = 1
            pos = 0     
            for i in range(1, len(new_token)):
                if new_token[i] == currentLetter:
                    currentConsecutive += 1
                    if i == len(new_token)-1:
                        if currentConsecutive > maxConsecutive:
                            maxConsecutive =  currentConsecutive
                            pos = i
                else:
                    if currentConsecutive > maxConsecutive:
                        maxConsecutive =  currentConsecutive
                        pos = i
                    currentLetter = new_token[i]
                    currentConsecutive = 1
            
            if maxConsecutive >= 2:
                new_token = new_token[:pos] + new_token[pos+1:]
        
            else:
                if new_token not in self.dictionary:
                    new_token = token
                break
            
        return new_token
            
            
        
    
    
########### TWEET PRE-PROCESSING ###########


# Reading tweets ---------------------------
# We are using Sentiment140 dataset with 1.600.000 tweets (no emoticons).

percentage_verbose = 10
p = 0.01

print("Reading tweets...")
tweets = pd.read_csv("tweets.csv", 
                        usecols=[0,5], sep=',', encoding="latin-1", names=["target", "text"], dtype=object,skiprows=lambda i: i>0 and random.random() > p)
numTweets = len(tweets)
print(str(numTweets) + " tweets read.")

tweets["processed"] = ""
tk = Tokenizer()

currentPERCENTAGE = percentage_verbose
step = math.floor((numTweets * percentage_verbose) / 100)

print("Processing tweets...")
for index in tweets.index: 
    
    tweet = tweets['text'][index]

    # Tokenization
    tokenized_tweet = tk.tokenize(tweet)
    tweets.at[index, 'processed'] = tokenized_tweet
    
    if index != 0 and index % step == 0:
        print("\t" + str(currentPERCENTAGE) + "% processed.")
        currentPERCENTAGE += percentage_verbose

print("Processing completed.\n")




########### TWEET CLASSIFICATION ###########

# Training models ---------------------------

# Divide data into training a test sets
print("Splitting data...\n")
tweets['processed'] = tweets['processed'].apply(lambda x: ' '.join([w for w in x]))
count_vectorizer = CountVectorizer(stop_words='english',binary=True) 
cv = count_vectorizer.fit_transform(tweets['processed'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(cv, tweets['target'], stratify=tweets['target'], test_size=0.2, random_state=54326)


# Classification ---------------------------

# Naive Bayes
start = time.time()
print("Calculating Naïve Bayes...")
clf = MultinomialNB()
clf.fit(x_train, y_train)
prediction_clf = clf.predict(x_test)
acc_clf = accuracy_score(prediction_clf,y_test)
end = time.time()
print("Accuracy = " + str(acc_clf) + "\n")
print("Time = " + str(end-start) + "\n")


# Decision Tree
start = time.time()
print("Calculating Decision Tree...")
dt = DecisionTreeClassifier(max_depth=100)
dt.fit(x_train,y_train)
prediction_dt = dt.predict(x_test)
acc_dt = accuracy_score(prediction_dt,y_test)
end = time.time()
print("Accuracy = " + str(acc_dt) + "\n")
print("Time = " + str(end-start) + "\n")


# Logistic Regression
start = time.time()
print("Calculating Logistic Regression...")
lr = LogisticRegression(verbose=True)
lr.fit(x_train,y_train)
prediction_lr = lr.predict(x_test)
acc_lr = accuracy_score(prediction_lr,y_test)
end = time.time()
print("Accuracy = " + str(acc_lr) + "\n")
print("Time = " + str(end-start) + "\n")


# SVM
start = time.time()
print("Calculating SVM...")
svc = svm.LinearSVC()
svc.fit(x_train,y_train)
prediction_svc = svc.predict(x_test)
acc_svc = accuracy_score(prediction_svc,y_test)
end = time.time()
print("Accuracy = " + str(acc_svc) + "\n")
print("Time = " + str(end-start) + "\n")




