
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Activation, Dropout, Dense, Input, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, precision_score, recall_score
import pandas as pd
import numpy as np
import string
import re
import os
import pyreadr

os.chdir("")

import helper_functions as hf

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

def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result

data['text']= data['text'].apply(lambda cw : remove_tags(cw))
data.head()

## Tokenize 
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

word_to_vec_map = read_glove_vector('glove_vectors_w2v.txt')

X_lengths = tokenizer.texts_to_sequences(data['text'])
maxLen = max([len(x) for x in X_lengths])

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
        tf.keras.layers.Dense(embed_vector_len, activation='gelu'),
        tf.keras.layers.Dense(4, activation='softmax')])
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
        print(f'FOLD {fold}')
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
        adam = tf.keras.optimizers.Adam(learning_rate = 0.0002)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        
        # Fit model
        model.fit(X_train_indices, Y_train, batch_size=64, epochs=15)

        # Test model
        X_test_indices = tokenizer.texts_to_sequences(X_test)
        X_test_indices = pad_sequences(X_test_indices, maxlen=maxLen, padding='post')
        model.evaluate(X_test_indices, Y_test)
        
        # Generate predictions
        preds = model.predict(X_test_indices)
        preds = pd.Series([np.argmax(x) for x in preds])
        preds.value_counts()

        ov_acc.append([accuracy_score(preds, Y_test), recall_score(preds, Y_test, average="macro"), precision_score(preds, Y_test, average="macro"),f1_score(preds, Y_test, average="macro")])
        f1.append(list(f1_score(Y_test,preds,average=None)))
        matrix = confusion_matrix(Y_test, preds)
        acc.append(list(matrix.diagonal()/matrix.sum(axis=1)))
        cr = pd.DataFrame(classification_report(Y_test,preds, output_dict=True)).transpose().iloc[0:4, 0:2]
        prec.append(list(cr.iloc[:,0]))
        rec.append(list(cr.iloc[:,1]))

lstm_glove_speeches_stats = []
lstm_glove_speeches_stats.append(
    {
        'Model': 'LSTM w/Glove',
        
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


import json
with open('results_speeches_LSTM' + '.txt', 'w') as outfile:
  json.dump(lstm_glove_speeches_stats, outfile)
