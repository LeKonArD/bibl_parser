import pandas as pd
import numpy as np
import re
import os
import keras
import torch
import flair
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Flatten, LSTM, Concatenate, concatenate,Reshape, multiply, GRU, Permute, merge, Bidirectional, Multiply, Lambda, RepeatVector
from keras.models import Model
from keras.metrics import categorical_accuracy
import keras.backend as K
from keras.utils import plot_model
from keras import regularizers
from keras.optimizers import Adadelta,Nadam,SGD,Adam
import keras
from keras.engine.topology import Layer
from collections import Counter
import torch
from keras import initializers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from flair.embeddings import ELMoEmbeddings, BertEmbeddings
from itertools import repeat

german_embedding = flair.embeddings.BertEmbeddings('bert-base-multilingual-cased')
def one_hot(x):
  if x == 0:
    return [1,0]
  else:
    return [0,1]
  
def gen_eval(sources, batch_size, seq_len):
  
  x_batch = []
  y_batch = []
  while True:
    
    for source in sources:
      
      data = pd.read_csv(source, sep="\t", index_col=0)
      num_samples = len(np.unique(data["sample"]))
      
      sample_pointer = int(len(np.unique(data["sample"]))/100*60)
    
      
      while sample_pointer <= num_samples:
        try:
           sample = data[data["sample"] == sample_pointer]
        
           y_label = list(sample["class"])[:seq_len+1]
           y_label = [one_hot(x) for x in y_label]
        
           while len(y_label) < seq_len:
             y_label.append([1,0])
           y_label = y_label[:seq_len]
        
           x_words = list(sample["text"].astype(str))[:seq_len+1]
        
           while len(x_words) < seq_len:
             x_words.append("<pad>")
           x_words = x_words[:seq_len]
        
           x_words = " ".join(x_words)
           x_words = flair.data.Sentence(x_words)
           print(x_words)


           german_embedding.embed(x_words)
           t_embed = []
           for token in x_words:
             t_embed.append(token.embedding.cpu().detach().numpy())
        
           x_batch.append(t_embed)
           y_batch.append(y_label)
           sample_pointer+=1   
           if len(x_batch) == batch_size:
          
             yield np.array(x_batch), np.array(y_batch)
        
             x_batch = []
             y_batch = []
        except RuntimeError:
           sample_pointer+=1

def gen_training(sources, batch_size, seq_len):
  
  x_batch = []
  y_batch = []
  while True:
    
    for source in sources:
    
      data = pd.read_csv(source, sep="\t", index_col=0)
      num_samples = int(len(np.unique(data["sample"]))/100*60)
      
      sample_pointer = 0
      while sample_pointer != num_samples:
         try:  
            sample = data[data["sample"] == sample_pointer]
        
            y_label = list(sample["class"])[:seq_len+1]
            y_label = [one_hot(x) for x in y_label]
        
            while len(y_label) < seq_len:
              y_label.append([1,0])
            y_label = y_label[:seq_len]
        
        
        
        
            x_words = list(sample["text"].astype(str))[:seq_len+1]
            
            while len(x_words) < seq_len:
              x_words.append("<pad>")
            x_words = x_words[:seq_len]
        
            x_words = " ".join(x_words)
            x_words = flair.data.Sentence(x_words)
        
            german_embedding.embed(x_words)
            t_embed = []
            for token in x_words:
              t_embed.append(token.embedding.cpu().detach().numpy())
            sample_pointer+=1       

        
            x_batch.append(t_embed)
            y_batch.append(y_label)
        
            if len(x_batch) == batch_size:
          
              yield np.array(x_batch), np.array(y_batch)
        
              x_batch = []
              y_batch = []
         except RuntimeError:
           sample_pointer+=1

def single_class_accuracy(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    accuracy_mask = K.cast(K.equal(class_id_preds, 1), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc
def declare_model():
  ## Input
  sample = Input(batch_shape=(10, 100, 3072))

  lstm_out1 = Bidirectional(GRU(32, return_sequences=True,dropout=0.5))(sample)
  lstm_out2 = Bidirectional(GRU(16, return_sequences=True, dropout=0.3))(lstm_out1)
  dense_out = Dense(8, activation='relu')(lstm_out2)
  predictions = Dense(2, activation='sigmoid')(dense_out)
  
  model = Model(inputs=sample, outputs=[predictions])
  model.compile(optimizer=Adam(lr=1e-3, clipnorm=4),
              loss='binary_crossentropy',
              metrics=[single_class_accuracy])
  print(model.summary())
  return model
m = declare_model()
m.fit_generator(gen_training(["./../gold/grobid_hum.tsv"], 10, 100), steps_per_epoch=6800/10, epochs=1)
m.save("grobid_authors.h5")

m.fit_generator(gen_training(["./../gold/dvjs_data.tsv"], 10, 100), steps_per_epoch=200/10, epochs=2)

m.save("dvjs_authors.h5")

print(m.evaluate_generator(gen_eval(["dvjs_data.tsv"], 10, 100), steps=10))
m.save("dvjs_authors.h5")
