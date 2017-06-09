
# coding: utf-8

# In[1]:

import json
import numpy
import logging
import cPickle as pickle

logging.getLogger().setLevel('INFO')
# In[2]:

logging.info("Loading alignments")
data = [ json.loads(line) for line in open("../data/coco/dataset.val.fa.json")]


# In[3]:

import imaginet.vendrov_provider as dp
import imaginet.defn.audiovis_rhn as audiovis


# In[4]:

logging.info("Loading audio features")
prov = dp.getDataProvider('coco', root='..', audio_kind='mfcc')


# In[5]:

val = list(prov.iterSentences(split='val'))


# In[6]:

def phones(utt):
    for word in utt['words']:
        pos = word['start']
        for phone in word['phones']:
            start = pos
            end = pos + phone['duration']
            pos = end
            label = phone['phone'].split('_')[0]
            if label != 'oov':
                yield (label, int(start*1000), int(end*1000)) 


# In[7]:

def slices(utt, rep, index=lambda ms: ms//10, aggregate=lambda x: x.mean(axis=0)):
    for phoneme in phones(utt):
        phone, start, end = phoneme
        assert index(start)<index(end)+1, "Something funny: {} {} {} {}".format(start, end, index(start), index(end))
        yield (phone, aggregate(rep[index(start):index(end)+1]))



# In[9]:

import imaginet.task as task


# In[10]:
logging.info("Loading RHN model")
model_rhn = task.load("../models/coco-speech.zip")


# In[ ]:

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# In[ ]:

logging.info("Extracting MFCC examples")
data_filter = [ (utt, sent) for (utt, sent) in zip(data, val) 
               if numpy.all([word.get('start', False) for word in utt['words']]) ]
data_filter = data_filter[:5000]
data_state =  [phoneme for (utt, sent) in data_filter for phoneme in slices(utt, sent['audio']) ]
y, X = zip(*data_state)
X = numpy.vstack(X)
y = numpy.array(y)
I = (X.shape[0]//3)*2
X_train = X[:I, :]
y_train = y[:I]
X_val = X[I:,:]
y_val = y[I:]

#scaler = PCA(n_components=13)
scaler = StandardScaler()
X_train_z = scaler.fit_transform(X_train)
X_val_z = scaler.transform(X_val)

logging.info("Training logistic regression")
#model = SGDClassifier(loss='log', random_state=123, n_iter=1)
model = LogisticRegression()
model.fit(X_train_z, y_train)
label = 'mfcc'
pred = model.predict(X_val_z)
pickle.dump(model, open("lr_{}.pkl".format(label), 'w'))
numpy.save('lr_{}_pred.npy'.format(label), numpy.vstack([y_val, pred]).T)
print(label, accuracy_score(y_val, pred))


logging.info("Extracting convo states")
states = audiovis.conv_states(model_rhn, [ sent['audio'] for utt,sent in data_filter ])
def index(t):
    return (t//10+6)//3
data_state =  [phoneme for i in range(len(data_filter)) for phoneme in slices(data_filter[i][0], states[i], index=index) ]
y, X = zip(*data_state)
X = numpy.vstack(X)
y = numpy.array(y)
I = (X.shape[0]//3)*2
X_train = X[:I, :]
y_train = y[:I]
X_val = X[I:,:]
y_val = y[I:]

#scaler = PCA(n_components=13)
scaler = StandardScaler()
X_train_z = scaler.fit_transform(X_train)
X_val_z = scaler.transform(X_val)
logging.info("Training logistic regression")
#model = SGDClassifier(loss='log', random_state=123, n_iter=1)
model = LogisticRegression()
model.fit(X_train_z, y_train)
label = 'conv'
pred = model.predict(X_val_z)
pickle.dump(model, open("lr_{}.pkl".format(label), 'w'))
numpy.save( 'lr_{}_pred.npy'.format(label), numpy.vstack([y_val, pred]).T)
print(label, accuracy_score(y_val, pred))


logging.info("Extracting recurrent layer states")
states = audiovis.layer_states(model_rhn, [ sent['audio'] for utt,sent in data_filter ])
def index(t):
    return (t//10+6)//3
def aggregate(x):
    return x[:,layer,:].mean(axis=0)

for layer in range(0,5):
    data_state =  [phoneme for i in range(len(data_filter)) for phoneme in slices(data_filter[i][0], states[i], index=index, aggregate=aggregate) ]
    y, X = zip(*data_state)
    X = numpy.vstack(X)
    y = numpy.array(y)
    I = (X.shape[0]//3)*2
    X_train = X[:I, :]
    y_train = y[:I]
    X_val = X[I:,:]
    y_val = y[I:]

    #scaler = PCA(n_components=13)
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_val_z = scaler.transform(X_val)
    logging.info("Training logistic regression for layer {}".format(layer))
    #model = SGDClassifier(loss='log', random_state=123, n_iter=1)
    model = LogisticRegression()
    model.fit(X_train_z, y_train)
    label = 'recurrent{}'.format(layer)
    pred = model.predict(X_val_z)
    pickle.dump(model, open("lr_{}.pkl".format(label), 'w'))
    numpy.save('lr_{}_pred.npy'.format(label), numpy.vstack([y_val, pred]).T)
    print(label, accuracy_score(y_val, pred))

