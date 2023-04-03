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

os.chdir("")

data = pd.read_excel("updated_coding_1219.xlsx")
len(data)
var = 'type' 
data[var].value_counts()
data = data[[var, "text"]]
data.loc[:, 'text'] = data['text'].str.replace('\n', ' ')
data = data.sample(frac=1).reset_index(drop=True)
data["label"] = data[var].astype('category')
data["label"] = data["label"].cat.codes

data['text'].dropna(inplace=True)
data['text'] = data['text'].astype(str)
data['text'] = [entry.lower() for entry in data['text']]

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

for i in range(0, 10):
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
        cr = pd.DataFrame(classification_report(Test_Y, predictions_SVM, output_dict=True)).transpose().iloc[0:3, 0:2]
        prec.append(list(cr.iloc[:,0]))
        rec.append(list(cr.iloc[:,1]))


civility_svm_stats = []
civility_svm_stats.append(
    {
        'Model': 'SVM',
        
        'civil_mean': np.mean([x[0] for x in acc]),
        'civil_mean_sd': np.std([x[0] for x in acc]),
        'civil_mean_f1': np.mean([x[0] for x in f1]),
        'civil_mean_f1_sd': np.std([x[0] for x in f1]),
        'civil_recall': np.mean([x[0] for x in rec]),
        'civil_recall_sd': np.std([x[0] for x in rec]),
        'civil_prec': np.mean([x[0] for x in prec]),
        'civil_prec_sd': np.std([x[0] for x in prec]),
        
        'incivil_mean': np.mean([x[1] for x in acc]),
        'incivil_mean_sd': np.std([x[1] for x in acc]),
        'incivil_mean_f1': np.mean([x[1] for x in f1]),
        'incivil_mean_f1_sd': np.std([x[1] for x in f1]),
        'incivil_recall': np.mean([x[1] for x in rec]),
        'incivil_recall_sd': np.std([x[1] for x in rec]),
        'incivil_prec': np.mean([x[1] for x in prec]),
        'incivil_prec_sd': np.std([x[1] for x in prec]),
        
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

with open('results_civility_SVM' + '.txt', 'w') as outfile:
  json.dump(civility_svm_stats, outfile)

