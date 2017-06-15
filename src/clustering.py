import imaginet.task as task
import imaginet.defn.audiovis_rhn as audiovis
import imaginet.data_provider as dp
import numpy
import json
import gzip
from itertools import izip
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from scipy.spatial.distance import euclidean
import cPickle as pickle

def getTimeStep(offset, model):
    return int((offset*100 + model.config['filter_length']) // model.config['stride'])

def getTimePoint(offset):
    return int(offset*100)


def getAudioVec(sentence, timepoint1, timepoint2):
    if timepoint1 == timepoint2:
        return sentence['audio'][timepoint1]
    return numpy.average(sentence['audio'][timepoint1:timepoint2],axis=0)

def getConvActivation(states, timestep1, timestep2):
    if timestep1 == timestep2:
        return states[timestep1]
    return numpy.average(states[timestep1:timestep2],axis=0)


def getActivation(states, layer, timestep1, timestep2):
    if timestep1 == timestep2:
        return states[timestep1][layer]
    return numpy.average(states[timestep1:timestep2,layer,:], axis=0)

def updatePhoneFreq(fvec, phone):
    if phone not in fvec: fvec[phone] = 0
    fvec[phone] += 1

def updateVector(vec, phone, newvec):
    if phone not in vec:
        vec[phone] = numpy.zeros(len(newvec))
    vec[phone] = numpy.add(vec[phone], newvec)

def normalizeVec(vec, freqs):
    for p in vec:
        x = [i/freqs[p] for i in vec[p]]
        vec[p] = numpy.asarray(x)
                                                                
def cluster(features, clusterno=2):
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=clusterno)
    clustering.fit(features)
    return clustering

def distanceMatrix(phones, vecs):
    pnum = len(phones)
    phonetab = []
    for i in range(pnum):
        for j in range(i+1):
            phonetab.append(euclidean(vecs[i],vecs[j]))
    return phonetab

def readAlignments():
    lines = list(gzip.open("../data/coco/dataset.val.fa.json.gz"))
    jdata = []
    for l in lines:
        jdata.append(json.loads(l))
    return jdata

def getFeatureVecs(jdata, model):
    prov = dp.getDataProvider('coco', root='..', audio_kind='mfcc', load_img=False)
    validate = prov.iterSentences(split='val')
    val_states=audiovis.iter_layer_states(model, [sent['audio'].astype('float32') for sent in prov.iterSentences(split='val')])
    conv_states = audiovis.iter_conv_states(model, [sent['audio'].astype('float32') for sent in prov.iterSentences(split='val')])

    layers = 5
    audios = {}
    convacts = {}
    activations = []
    for l in range(layers): activations.append({})
    freqs = {}

    for sent, vstates, cstates, ali in izip(validate, val_states, conv_states, jdata):
        for word in ali['words']:
            if word['case'] != 'success': continue
        
            curser = word['start']
            for p in word['phones']:
                phone = p['phone'].split('_')[0]
                duration = p['duration']
                
                timepoint1 = getTimePoint(curser)
                timepoint2 = getTimePoint(curser+duration)
                audiovec = getAudioVec(sent, timepoint1, timepoint2)
                updateVector(audios, phone, audiovec)
                
                timestep1 = getTimeStep(curser, model)
                timestep2 = getTimeStep(curser+duration, model)
                convvec = getConvActivation(cstates, timestep1, timestep2)
                updateVector(convacts, phone, convvec)
                
                for layer in range(layers):
                    activevec = getActivation(vstates, layer, timestep1, timestep2)
                    updateVector(activations[layer], phone, activevec)
                    
                updatePhoneFreq(freqs, phone)
                curser += duration

    #normalize vectors
    for l in range(layers): normalizeVec(activations[l], freqs)
    normalizeVec(audios, freqs)
    normalizeVec(convacts, freqs)
    audios.pop('oov', None)
    convacts.pop('oov', None)
    for i in range(layers): activations[i].pop('oov', None)
    return audios,convacts,pactivations
    
def phoneme_correlation():
    model = task.load("../models/coco-speech.zip")
    alignments = readAlignments()
    audios,convacts,pactivations = getFeatureVecs(alignments,model)
    phones = list(audios)
    paudios = [audios[p] for p in phones]
    pconvacts = [convacts[p] for p in phones]
    pactivations = {}
    for l in range(layers):
        pactivations[l] = [activations[l][p] for p in phones]

    out = open('pearsonr.csv','w')
    out.write('Representation r\n')
    audio_matrix = distanceMatrix(phones, paudios)
    convact_matrix = distanceMatrix(phones, pconvacts)
    out.write("conv %2.2f\n"%numpy.corrcoef(audio_matrix, convact_matrix)[0][1])
    for l in range(layers):
        activation_matrix = distanceMatrix(phones, pactivations[l])
        out.write("rec%d %2.2f\n"%(l,numpy.corrcoef(audio_matrix, activation_matrix)[0][1]))
    out.close()
