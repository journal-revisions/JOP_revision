
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Activation, Dropout, Dense, Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, precision_score, recall_score
import pandas as pd
import numpy as np
import string
import re
import os

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

def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result

data['text']= data['text'].apply(lambda cw : remove_tags(cw))
data.head()

# Tokenize 
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data["text"])
words_to_index = tokenizer.word_index
len(words_to_index) ## keeping all words

def read_glove_vector(glove_vec):
    with open(glove_vec, 'r', errors = 'ignore', encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            w_line = line.split(' ')
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)
    return word_to_vec_map

word_to_vec_map = read_glove_vector('Baseline models/glove_vectors_w2v.txt')

X_all = tokenizer.texts_to_sequences(data["text"])
maxLen = max([len(x) for x in X_all])  

vocab_len = len(words_to_index)
embed_vector_len = word_to_vec_map['moon'].shape[0]

emb_matrix = np.zeros((vocab_len+1, embed_vector_len))

for word, index in words_to_index.items():
    embedding_vector = word_to_vec_map.get(word)
    if embedding_vector is not None:
        emb_matrix[index, :] = embedding_vector

## Define general BiLSTM RNN MODEL:
def lstm_model(vocab_len, embed_vector_len):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_len, embed_vector_len, weights = [emb_matrix]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embed_vector_len)),
        tf.keras.layers.Dense(embed_vector_len, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')])
    return model 

## 10-fold Cross-Validation: 

ov_acc = []
f1 = []
acc = []
prec = []
rec = []

for i in range(1, 10):
    kfold = KFold(n_splits=10, shuffle=True)    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
        print(f'FOLD {fold+1}')
        train = data.iloc[train_ids,]
        test = data.iloc[test_ids,]
        X_train = pd.Series(train["text"])
        Y_train = pd.Series(train["label"])
        X_test = pd.Series(test["text"])
        Y_test = pd.Series(test["label"])
        #Y_train = np.asarray(Y_train).astype('float32').reshape((-1,1))
        #Y_test = np.asarray(Y_test).astype('float32').reshape((-1,1))
        
        ## TRAINING X indices
        X_train_indices = tokenizer.texts_to_sequences(X_train)
        X_train_indices = pad_sequences(X_train_indices, maxlen=maxLen, padding='post')
        X_train_indices.shape
        
        # Start model
        model = lstm_model(vocab_len+1, embed_vector_len)
        adam = tf.keras.optimizers.Adam(learning_rate = 0.0001)
        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
        
        # Fit model
        model.fit(X_train_indices, Y_train, batch_size=64, epochs=15)

        # Test model
        X_test_indices = tokenizer.texts_to_sequences(X_test)
        X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen, padding='post')
        model.evaluate(X_test_indices, Y_test)
        
        # Generate predictions
        preds = model.predict(X_test_indices)
        preds = pd.Series([int(np.round(x)) for x in preds])
        preds.value_counts()

        ov_acc.append([accuracy_score(preds, Y_test), recall_score(preds, Y_test, average="macro"), precision_score(preds, Y_test, average="macro"),f1_score(preds, Y_test, average="macro")])
        f1.append(list(f1_score(Y_test,preds,average=None)))
        matrix = confusion_matrix(Y_test, preds)
        acc.append(list(matrix.diagonal()/matrix.sum(axis=1)))
        cr = pd.DataFrame(classification_report(Y_test,preds, output_dict=True)).transpose().iloc[0:3, 0:2]
        prec.append(list(cr.iloc[:,0]))
        rec.append(list(cr.iloc[:,1]))



lstm_glove_civility_stats = []
lstm_glove_civility_stats.append(
    {
        'Model': 'LSTM w/GloVe',
        'Variable': var,

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

import json
with open('results_civility_LSTM' + '.txt', 'w') as outfile:
  json.dump(lstm_glove_civility_stats, outfile)
















