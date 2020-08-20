import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import pandas as pd
import numpy as np
import re
import torch
import flair
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from flair.embeddings import ELMoEmbeddings, BertEmbeddings
flair.device = torch.device('cpu')
from itertools import repeat
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import tensorflow.keras.backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = '-1' 
german_embedding = flair.embeddings.BertEmbeddings('bert-base-multilingual-cased')

def one_hot(x):
  if x == 0:
    return [1,0]
  else:
    return [0,1]

def single_class_accuracy(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    accuracy_mask = K.cast(K.equal(class_id_preds, 1), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc

def gen_prediction(sources, batch_size, seq_len):
  
    x_batch = []
 
    for bib in sources:
        x_words = bib.split(" ")
        x_words = [x for x in x_words if x != ""]

        while len(x_words) < seq_len:
            x_words.append("<pad>")
        x_words = x_words[:seq_len]

        x_words = " ".join(x_words)
        x_words = flair.data.Sentence(x_words)

        german_embedding.embed(x_words)
        t_embed = []
        for token in x_words:
            t_embed.append(token.embedding.cpu().detach().numpy())

        x_batch.append(t_embed)

        if len(x_batch) == batch_size:

            yield np.array(x_batch)

            x_batch = []
            
dvjs = pd.read_csv("./../data/dvjs_all_bibl.tsv", index_col=0, sep="\t")
inputs = list(dvjs["text"])
gen = gen_prediction(inputs,10,100)
m = load_model("./../models/dvjs_authors.h5", custom_objects={"single_class_accuracy":single_class_accuracy})

prediction = []
i = 0
for s in tqdm(gen):

    prediction.append(np.argmax(m.predict(s), axis=-1))
    
    i += 1

output = [x for y in prediction for x in y]

output = np.array(output)

import pickle as pkl


with open("dvjs_pred.pkl", "wb") as f:
    
    pkl.dump(output,f)




