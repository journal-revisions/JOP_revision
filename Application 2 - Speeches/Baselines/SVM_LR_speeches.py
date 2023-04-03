import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import sklearn
import spacy
import os
import json
from tqdm import trange
import pyreadr

os.chdir("")

data = pyreadr.read_r("speeches_translated.RData") # use 
print(data.keys())
data = data["dataset"]
data

data_labels = pd.read_csv("speeches_and_scores_label.csv") # use 
data_labels = pd.DataFrame(data_labels["speechtype"])
data_labels.set_index(data.index, inplace=True)
data_labels.value_counts()
data["text"] = [i.replace("\n", " ") for i in data["text"]]

data = pd.concat([data, data_labels], axis=1)
data['speechtype'].value_counts()
data["label"] = data['speechtype'].astype('category')
data["label"] = data["label"].cat.codes
data = data.sample(frac=1).reset_index(drop=True)
data["label"].value_counts()

data['text'].dropna(inplace=True)
data['text'] = data['text'].astype(str)
data['text'] = [entry.lower() for entry in data['text']]


## TRANSLATE EVERYTHING TO ENGLISH FIRST FOR BASELINE CLASSIFIERS (use free GTranslate code API)
data['text'] = [word_tokenize(entry) for entry in data['text']]
for index,entry in enumerate(data['text']):
    Final_words = []
    word_Final = [t for t in entry if not t in stopwords.words("english")]
    Final_words.append(word_Final)
    data.loc[index,'text_final'] = str(Final_words)


## CV SVM
ov_acc = []
f1 = []
acc = []
prec = []
rec = []

for i in range(0, 3):
    kfold = KFold(n_splits=10, shuffle=True)    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
        print(f'FOLD {fold}')
        train = data.iloc[train_ids,]
        test = data.iloc[test_ids,]
        Train_X = pd.Series(train["text_final"])
        Train_Y = pd.Series(train["label"])
        Test_X = pd.Series(test["text_final"])
        Test_Y = pd.Series(test["label"])
        Encoder = LabelEncoder()
        Train_Y = Encoder.fit_transform(Train_Y)
        Test_Y = Encoder.fit_transform(Test_Y)
        Tfidf_vect = TfidfVectorizer(max_features=5000)
        Tfidf_vect.fit(train["text_final"])
        Train_X_Tfidf = Tfidf_vect.transform(Train_X)
        Test_X_Tfidf = Tfidf_vect.transform(Test_X)

        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        SVM.fit(Train_X_Tfidf, Train_Y)
        predictions_SVM = SVM.predict(Test_X_Tfidf)
        
        #print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
        ov_acc.append([accuracy_score(predictions_SVM, Test_Y), recall_score(predictions_SVM, Test_Y, average="macro"), precision_score(predictions_SVM, Test_Y, average="macro"),f1_score(predictions_SVM, Test_Y, average="macro")])
        f1.append(list(f1_score(Test_Y,predictions_SVM,average=None)))
        matrix = sklearn.metrics.confusion_matrix(Test_Y, predictions_SVM)
        acc.append(list(matrix.diagonal()/matrix.sum(axis=1)))
        cr = pd.DataFrame(classification_report(Test_Y, predictions_SVM, output_dict=True)).transpose().iloc[0:4, 0:2]
        cr
        prec.append(list(cr.iloc[:,0]))
        rec.append(list(cr.iloc[:,1]))


speeches_svm_stats = []
speeches_svm_stats.append(
    {
        'Model': 'SVM',
        
        'campaign_mean': np.mean([x[0] for x in acc]),
        'campaign_mean_sd': np.std([x[0] for x in acc]),
        'campaign_mean_f1': np.mean([x[0] for x in f1]),
        'campaign_mean_f1_sd': np.std([x[0] for x in f1]),
        'campaign_recall': np.mean([x[0] for x in rec]),
        'campaign_recall_sd': np.std([x[0] for x in rec]),
        'campaign_prec': np.mean([x[0] for x in prec]),
        'campaign_prec_sd': np.std([x[0] for x in prec]),
        
        'famous_mean': np.mean([x[1] for x in acc]),
        'famous_mean_sd': np.std([x[1] for x in acc]),
        'famous_mean_f1': np.mean([x[1] for x in f1]),
        'famous_mean_f1_sd': np.std([x[1] for x in f1]),
        'famous_recall': np.mean([x[1] for x in rec]),
        'famous_recall_sd': np.std([x[1] for x in rec]),
        'famous_prec': np.mean([x[1] for x in prec]),
        'famous_prec_sd': np.std([x[1] for x in prec]),
        
        'international_mean': np.mean([x[2] for x in acc]),
        'international_mean_sd': np.std([x[2] for x in acc]),
        'international_mean_f1': np.mean([x[2] for x in f1]),
        'international_mean_f1_sd': np.std([x[2] for x in f1]),
        'international_recall': np.mean([x[2] for x in rec]),
        'international_recall_sd': np.std([x[2] for x in rec]),
        'international_prec': np.mean([x[2] for x in prec]),
        'international_prec_sd': np.std([x[2] for x in prec]),

        'ribboncutting_mean': np.mean([x[3] for x in acc]),
        'ribboncutting_mean_sd': np.std([x[3] for x in acc]),
        'ribboncutting_mean_f1': np.mean([x[3] for x in f1]),
        'ribboncutting_mean_f1_sd': np.std([x[3] for x in f1]),
        'ribboncutting_recall': np.mean([x for x in rec]),
        'ribboncutting_recall_sd': np.std([x[3] for x in rec]),
        'ribboncutting_prec': np.mean([x[3] for x in prec]),
        'ribboncutting_prec_sd': np.std([x[3] for x in prec]),        
        
        'overall_mean': np.mean([x[0] for x in ov_acc]),
        'overall_mean_sd': np.std([x[0] for x in ov_acc]),
        'overall_mean_f1': np.mean([x[3] for x in ov_acc]),
        'overall_mean_f1_sd': np.std([x[3] for x in ov_acc]),
        'overall_recall': np.mean([x[1] for x in ov_acc]),
        'overall_recall_sd': np.std([x[1] for x in ov_acc]),
        'overall_prec': np.mean([x[2] for x in ov_acc]),
        'overall_prec_sd': np.std([x[2] for x in ov_acc]),
    }
)  

with open('results_speeches_SVM' + '.txt', 'w') as outfile:
  json.dump(speeches_svm_stats, outfile)
