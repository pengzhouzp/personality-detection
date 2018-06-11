
# coding: utf-8

# In[92]:


import keras
import numpy as np
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.layers.embeddings import Embedding
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import nltk
#from nltk.corpus import stopwords
#from nltk.stem import SnowballStemmer
import re
import string
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.cross_validation import train_test_split
from tensorflow.contrib import learn
import os
from collections import defaultdict
from keras.callbacks import EarlyStopping
#from gensim import corpora


# In[3]:


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s ", string)
    string = re.sub(r"\'ve", " have ", string)
    string = re.sub(r"n\'t", " not ", string)
    string = re.sub(r"\'re", " are ", string)
    string = re.sub(r"\'d" , " would ", string)
    string = re.sub(r"\'ll", " will ", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
#   string = re.sub(r"[^\d\.]","",string)
#   string = re.sub(r"[a-zA-Z]{4,}", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


# In[4]:


def get_rawdata(filename):
    df=pd.read_csv(filename,encoding='latin-1')
    raw=df['TEXT']
    EXT=pd.get_dummies(df['cEXT'])['y'].as_matrix()
    NEU=pd.get_dummies(df['cNEU'])['y'].as_matrix()
    AGR=pd.get_dummies(df['cAGR'])['y'].as_matrix()
    CON=pd.get_dummies(df['cCON'])['y'].as_matrix()
    OPN=pd.get_dummies(df['cOPN'])['y'].as_matrix()
    EXT=EXT.reshape(len(EXT),1)
    NEU=NEU.reshape(len(NEU),1)
    AGR=AGR.reshape(len(AGR),1)
    CON=CON.reshape(len(CON),1)
    OPN=OPN.reshape(len(OPN),1)
    ytrain=np.concatenate((EXT,NEU,AGR,CON,OPN),axis=1)
    return raw,ytrain
    

    


# In[5]:


def get_chargedwords(filename):
    df=pd.read_csv(filename,encoding='latin-1')
    emontial=df[df['Charged']!=0]['Words'].tolist()
    return emontial
    


# In[6]:


def clean_essay(raw,emontial):
    doc=[]
    sentence_length=[]
    for i in raw:
        essay=re.split(r'[.?!]',i)
        sentences=[]
        for sent in essay:
            sent=clean_str(sent)
            sent=sent.split(' ')
            if(len(sent)>150):
                newsent=[]
                splits=int(np.floor(len(sent)/20))
                for index in range(splits):
                            newsent.append(' '.join(sent[index*20:(index+1)*20]))
                if len(sent)>splits*20:
                            newsent.append(' '.join(sent[splits*20:]))
                for s in newsent:
                    n=0
                    s=s.split(' ')
                    for word in s:
                        if word in emontial:
                            n=n+1
                    if n!=0:
                        sentence_length.append(len(s))
                        s=' '.join(s)
                        sentences.append(s)
                    
            else:
                n=0
                for word in sent:
                    if word in emontial:
                        n=n+1
                if n!=0:
                    sentence_length.append(len(sent))
                    sent=' '.join(sent)
                    sentences.append(sent)    
        doc.append(sentences)
    return doc,max(sentence_length)


# In[7]:


def merge_sents(doc):
    b=[]
    for essay in doc:
        for sent in essay:
            b.append(sent)
    return b


# In[82]:


def get_meairesse(filename):
    df1=pd.read_csv(filename,encoding='latin-1',header=None)
    df2=pd.read_csv('essays.csv',encoding='latin-1')
    df1=df1.set_index(df1.iloc[:,0])
    df1=df1.reindex(index=df2.iloc[:,0])
    ma=df1.iloc[:,1:].as_matrix()
    return ma


# In[9]:
path=os.path.abspath('..')
print(path)
[raw,labels]=get_rawdata('essays.csv')

emontial=get_chargedwords('Emotion_Lexicon.csv')

[doc,max_wordNum]=clean_essay(raw,emontial)

merged_essay=merge_sents(doc)



# In[83]:


ma=get_meairesse('mairesse.csv')

# # 1. Get word-embedding matrix from pre-trained model

# In[38]:

vocab_processor = learn.preprocessing.VocabularyProcessor(max_wordNum)
vocab_processor.fit_transform(merged_essay)
vocab_dict = vocab_processor.vocabulary_._mapping


# In[39]:


def pre_trained_embedding(dictionary,filename):

    embeddings_index = {}
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    #print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(dictionary), 300))
    for word, i in dictionary.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# In[48]:

#glove_path=os.path.join(path,'glove.6B.300d.txt')
#print(glove_path)
M_pre_train=pre_trained_embedding(vocab_dict,'glove.6B.300d.txt')


# In[41]:


#convert text to martix
def text_to_martix(doc):
    data=doc.copy()
    for i in range(len(data)):
        data[i]=np.array(list(vocab_processor.transform(data[i])))
    data=np.array(data)
    numsentences=[]
    for i in range(len(data)):
        numsentences.append(len(data[i]))
    max_sentNum=max(numsentences)
    for i in range(len(data)):
        ess=data[i]
        if len(ess)<max_sentNum:
            zeros=np.zeros([max_sentNum-len(ess),max_wordNum])
            data[i]=np.concatenate((ess,zeros),axis=0)
    return data,max_sentNum


# In[50]:


[data,max_sentNum]=text_to_martix(doc)


# In[51]:


def data_reshape(data):
    y=np.dstack(data)
    y=np.rollaxis(y,-1)
    return y
#return a martix whose shape is (# of essays,# of sentences,# of words,)


# In[84]:


xtrain=data_reshape(data)
X_train,X_test,y_train,y_test,ma_train,ma_test=train_test_split(xtrain,labels,ma,test_size=0.3)


'''
embedding_vecor_length = 300
model_main = Sequential()
model_main.add(Embedding(len(vocab_dict),embedding_vecor_length,weights=[M_pre_train]
                       ,input_shape=(max_sentNum,max_wordNum),trainable=True,name='Embedding_1'))

model_main.add(Reshape((max_sentNum, max_wordNum , embedding_vecor_length , 1) , name='Reshape_1'))

model_1gram = Sequential()
model_1gram.add(TimeDistributed(Conv2D(200, kernel_size=(1, embedding_vecor_length), activation='relu' , use_bias=True) , input_shape = (max_sentNum, max_wordNum , embedding_vecor_length , 1) , name='1Gram_TimeDistributed_Conv1'))
model_1gram.add(TimeDistributed(MaxPooling2D(pool_size=(max_wordNum - 1 + 1, 1)) , name='1Gram_TimeDistributed_Maxpool1'))

model_2gram = Sequential()
model_2gram.add(TimeDistributed(Conv2D(200, kernel_size=(2, embedding_vecor_length), activation='relu' , use_bias=True) , input_shape = (max_sentNum, max_wordNum , embedding_vecor_length , 1) , name='2Gram_TimeDistributed_Conv1'))
model_2gram.add(TimeDistributed(MaxPooling2D(pool_size=(max_wordNum - 2 + 1, 1)) , name='2Gram_TimeDistributed_Maxpool1'))

model_3gram = Sequential()
model_3gram.add(TimeDistributed(Conv2D(200, kernel_size=(3, embedding_vecor_length), activation='relu' , use_bias=True) , input_shape = (max_sentNum, max_wordNum , embedding_vecor_length , 1) , name='3Gram_TimeDistributed_Conv1'))
model_3gram.add(TimeDistributed(MaxPooling2D(pool_size=(max_wordNum - 3 + 1, 1)) , name='3Gram_TimeDistributed_Maxpool1'))

model_ngram = Sequential()
merged = Merge([model_1gram,model_2gram,model_3gram], mode='concat',name='Merge_n-grams')
model_main.add(merged)
'''
from keras.models import Model
from keras.utils import plot_model
from keras.layers.merge import concatenate


embedding_vecor_length = 300


x = Input(shape=(max_sentNum, max_wordNum))


embedding_layer = Embedding(len(vocab_dict),embedding_vecor_length,weights=[M_pre_train]
                       ,input_shape=(max_sentNum,max_wordNum),trainable=True,name='Embedding')(x)


reshape_layer = Reshape((max_sentNum, max_wordNum , embedding_vecor_length , 1) , name='Reshape_Embedding')(embedding_layer)


one_gram_conv = TimeDistributed(Conv2D(200, kernel_size=(1, embedding_vecor_length), activation='relu' , use_bias=True) 
                              , name='1Gram_TimeDistributed_Conv')(reshape_layer)
one_gram_maxpool = TimeDistributed(MaxPooling2D(pool_size=(max_wordNum - 1 + 1, 1)) 
                                   , name='1Gram_TimeDistributed_Maxpool')(one_gram_conv)
one_gram_flatten = TimeDistributed(Flatten() , name='1Gram_TimeDistributed_Flatten')(one_gram_maxpool)


two_gram_conv = TimeDistributed(Conv2D(200, kernel_size=(2, embedding_vecor_length), activation='relu' , use_bias=True) 
                                , name='2Gram_TimeDistributed_Conv')(reshape_layer)
two_gram_maxpool = TimeDistributed(MaxPooling2D(pool_size=(max_wordNum - 2 + 1, 1)) 
                                   , name='2Gram_TimeDistributed_Maxpool')(two_gram_conv)
two_gram_flatten = TimeDistributed(Flatten() , name='2Gram_TimeDistributed_Flatten')(two_gram_maxpool)


three_gram_conv = TimeDistributed(Conv2D(200, kernel_size=(3, embedding_vecor_length), activation='relu' , use_bias=True) 
                                  , name='3Gram_TimeDistributed_Conv')(reshape_layer)
three_gram_maxpool = TimeDistributed(MaxPooling2D(pool_size=(max_wordNum - 3 + 1, 1)) 
                                , name='3Gram_TimeDistributed_Maxpool')(three_gram_conv)
three_gram_flatten = TimeDistributed(Flatten() , name='3Gram_TimeDistributed_Flatten')(three_gram_maxpool)


merge = concatenate([one_gram_flatten, two_gram_flatten , three_gram_flatten] , name='Merge_n-grams')


reshape_merge = Reshape((max_sentNum , 600 , 1) , name='Reshape_Merge_Result')(merge)

one_max_pool = MaxPooling2D(pool_size=(max_sentNum, 1) , name='1-MaxPool')(reshape_merge)

one_max_pool_flatten = Flatten(name='1-Maxpool_Flatten')(one_max_pool)


mairesse_input = Input(shape=(84,) , name='Mairesse_Input')
document_merge = concatenate([one_max_pool_flatten , mairesse_input] , name='Merge_with_Mairesse')


fully_connected = Dense(16,activation='sigmoid',name='Fully_Connected_Layer')(document_merge)
classification_layer = Dense(5, activation='sigmoid',name='Dense_out')(fully_connected)


output = classification_layer
model = Model(inputs=[x , mairesse_input] , outputs=output)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
#plot_model(model, to_file='shared_input_layer.png')

callbacks=[EarlyStopping(monitor='val_loss',patience=2,verbose=0)]

history=model.fit([X_train,ma_train],y_train,validation_data=[[X_test,ma_test],y_test],epochs=30, batch_size=200 , callbacks=callbacks)

model.save('cnn.h5')

import matplotlib.pyplot as plt

fig=plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('train loss vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper right')
fig.savefig('cnn.png')

