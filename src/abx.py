import numpy
from audio import load_audio, extract_mfcc
from activations import save_activations
from sets import Set
from scipy.spatial.distance import euclidean

datadir = "../data/coco/"

def minimum_pairs(syllable, all_syllables):
        mps = Set()
        syllable = syllable.split('_')
        for syl in all_syllables.difference(syllable):
                s = syl.split('_')
                if ((s[0]==syllable[0] and s[1]!=syllable[1]) or (s[1]==syllable[1] and s[0]!=syllable[0])):
                        mps.add(syl)
        return mps

def prepareItems(all_syllables):
        items = []
        for a in all_syllables:
                a_mp = minimum_pairs(a, all_syllables)
                for b in a_mp:
                        xlist = minimum_pairs(b, all_syllables).difference(a_mp).difference([a])
                        for x in xlist:
                                items.append((a,b,x))
        return items
                                                                                                                                    
def audiovec(item, features):
        return numpy.average(features[item],axis=0)

def embvec(item, features):
        return features[item]

def convvec(item, features):
        return numpy.average(features[item][:], axis=0)

def layervec(item, layer, features):
        return numpy.average(features[item][:,layer], axis=0)

def distance(a,b):
        return euclidean(a,b)
                    
def abx_all():
    lines = [line.strip() for line in open("abx_cv.txt")][1:]
    words = [l.split()[-1] for l in lines if l[0] != '#']
    syllables = [l.lower().split()[0]+'_'+l.lower().split()[1] for l in lines if l[0] != '#']
    all_syllables = Set(syllables)
    items = prepareItems(all_syllables)

    audios = list(load_audio(words, datadir+"abx_cv/"))
    save_activations(audios, "../models/coco-speech.zip", datadir+"abx-mfcc.npy", datadir+"abx-conv_states.npy", \
                     datadir+"abx-states.npy", datadir+"abx-embeddings.npy")

    out = open("abx-all.txt", 'w')

    f = dict(zip(syllables, [extract_mfcc(a) for a in audios]))
    diff = [distance(audiovec(a,f),audiovec(x,f)) - distance(audiovec(b,f),audiovec(x,f)) for (a,b,x) in items]
    hits = [int(i>=0) for i in diff]
    out.write("mfcc,%2.2f\n"%(float(sum(hits))/len(hits)))
    
    f = dict(zip(syllables,numpy.load(datadir+"abx-conv_states.npy")))
    diff = [distance(convvec(a,f),convvec(x,f)) - distance(convvec(b,f),convvec(x,f)) for (a,b,x) in items]
    hits = [int(i>=0) for i in diff]
    out.write("convolution,%2.2f\n"%(float(sum(hits))/len(hits)))

    f = numpy.load(datadir+"abx-states.npy")
    layers = len(f[0][0])
    f = dict(zip(syllables, f))
    for l in range(layers):
            diff = [distance(layervec(a,l,f),layervec(x,l,f)) - distance(layervec(b,l,f),layervec(x,l,f)) for (a,b,x) in items]
            hits = [int(i>=0) for i in diff]
            out.write("recurrent %d,%2.2f\n"%(l,float(sum(hits))/len(hits)))

    f = dict(zip(syllables,numpy.load(datadir+"abx-embeddings.npy")))
    diff = [distance(embvec(a,f),embvec(x,f)) - distance(embvec(b,f),embvec(x,f)) for (a,b,x) in items]
    hits = [int(i>=0) for i in diff]
    out.write("embeddings,%2.2f\n"%(float(sum(hits))/len(hits)))

    out.close()

    
def minPhone(x,y):
        x = x.split('_')
        y = y.split('_')
        if (x[0]==y[0] and x[1]!=y[1]):
                return x[0]
        if (x[1]==y[1] and x[0]!=y[0]):
                return x[1]
        return ''

def phonemeClass(phoneme, phoneme_classes, gold_classes):
        if phoneme not in phoneme_classes: return 'unknown'
        return gold_classes[phoneme_classes[phoneme]]

def getClassDiffs(items, diffs, phoneme_classes, gold_classes):
        classdiff = {}
        for c in gold_classes + ['unknown']: classdiff[c] = []
        for i in range(len(items)):
                (a,b,x) = items[i]
                classdiff[phonemeClass(minPhone(b,x), phoneme_classes, gold_classes)].append(diffs[i])
        return classdiff

def getClassAccuracies(classdiff, gold_classes):
        accuracies = []
        for c in gold_classes:
                hits = [int(i>0) for i in classdiff[c]]
                if len(hits) == 0: hits.append(0.0)
                accuracies.append(float(sum(hits))/len(hits))
        return accuracies


def getClasses():
        lines = list(open('phonemes.txt'))[1:]
        gold_classes = []
        phoneme_classes = {}
        ipa = {}

        for line in lines:
                cols = line.strip().split('\t')
                if cols[2] not in gold_classes: gold_classes.append(cols[2])
                phoneme_classes[cols[0]] = gold_classes.index(cols[2])
                ipa[cols[0]] = cols[1]
        return phoneme_classes,gold_classes,ipa

def abx_classes():
        phoneme_classes,gold_classes,ipa = getClasses()
        lines = [line.strip() for line in open("abx_cv.txt")][1:]
        words = [l.split()[-1] for l in lines if l[0] != '#']
        syllables = [l.lower().split()[0]+'_'+l.lower().split()[1] for l in lines if l[0] != '#']
        all_syllables = Set(syllables)
        items = prepareItems(all_syllables)
        
        audios = list(load_audio(words, datadir+"abx_cv/"))
        save_activations(audios, "../models/coco-speech.zip", datadir+"abx-mfcc.npy", datadir+"abx-conv_states.npy", \
                         datadir+"abx-states.npy", datadir+"abx-embeddings.npy")
        
        out = open("abx-classes.csv", 'w')
        
        out.write("Phoneme class,"+(''.join("%s\t"%c for c in gold_classes)))

        f = dict(zip(syllables, [extract_mfcc(a) for a in audios]))
        diff = [distance(audiovec(a,f),audiovec(x,f)) - distance(audiovec(b,f),audiovec(x,f)) for (a,b,x) in items]
        accuracies = getClassAccuracies(getClassDiffs(items, diff, phoneme_classes, gold_classes),gold_classes)
        out.write("mfcc,"+(''.join("%2.2f\t"%accuracies[c] for c in range(len(gold_classes)))))

        f = dict(zip(syllables,numpy.load(datadir+"abx-conv_states.npy")))
        diff = [distance(convvec(a,f),convvec(x,f)) - distance(convvec(b,f),convvec(x,f)) for (a,b,x) in items]
        accuracies = getClassAccuracies(getClassDiffs(items, diff, phoneme_classes, gold_classes),gold_classes)
        out.write("convolution,"+(''.join("%2.2f\t"%accuracies[c] for c in range(len(gold_classes)))))

        f = numpy.load(datadir+"abx-states.npy")
        layers = len(f[0][0])
        f = dict(zip(syllables, f))
        for l in range(layers):
                diff = [distance(layervec(a,l,f),layervec(x,l,f)) - distance(layervec(b,l,f),layervec(x,l,f)) for (a,b,x) in items]
                accuracies = getClassAccuracies(getClassDiffs(items, diff, phoneme_classes, gold_classes),gold_classes)
                out.write("recurrent %d,"%l+(''.join("%2.2f\t"%accuracies[c] for c in range(len(gold_classes)))))

        f = dict(zip(syllables,numpy.load(datadir+"abx-embeddings.npy")))
        diff = [distance(embvec(a,f),embvec(x,f)) - distance(embvec(b,f),embvec(x,f)) for (a,b,x) in items]
        accuracies = getClassAccuracies(getClassDiffs(items, diff, phoneme_classes, gold_classes),gold_classes)
        out.write("embeddings,"+(''.join("%2.2f\t"%accuracies[c] for c in range(len(gold_classes)))))
                    
        out.close()
                                                                                                                                                        
