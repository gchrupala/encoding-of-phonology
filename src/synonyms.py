import numpy
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
import logging

pairs = ["make/prepare",
    "slice/cut",
    "person/someone",
    "photo/picture",
    "picture/image",
    "kid/child",
    "photograph/picture",
    "slice/piece",
    "bicycle/bike",
    "photograph/photo",
    "couch/sofa",
    "tv/television",
    "vegetable/veggie",
    "sidewalk/pavement",
    "rock/stone",
    "store/shop",
    "purse/bag",
    "direction/way",
    "assortment/variety",
    "spot/place",
    "pier/dock",
    "carpet/rug",
    "bun/roll",
    "large/big",
    "small/little" ]

def split_into_pairs(arr):
    index = [
        [0, 132],
        [132,392],
        [392,2662],
        [2662,3874],
        [3874,5000],
        [5000,6124],
        [6124,7000],
        [7000,7732],
        [7732,8392],
        [8392,8978],
        [8978,9520],
        [9520,9976],
        [9976,10390],
        [10390,10844],
        [10844,11210],
        [11210,11512],
        [11512,11778],
        [11778,11888],
        [11888,12020],
        [12020,12138],
        [12138,12256],
        [12256,12358],
        [12358,12454],
        [12454,14736],
        [14736,16804]]
    for ix in index:
        yield arr[ix[0]:ix[1]]


def create_labels(array):
    labels = []
    for i in range(int(array.shape[0]/2)):
        labels.append(0)
        labels.append(1)
    return labels

def timewise_average(array):
    layer = []
    for element in array:
        c = numpy.mean(element, axis =0)
        layer.append(c)
    data = numpy.array(layer)
    return data

def layer_activation(array, layer):
    activations = []
    for element in array:
        b = numpy.array(element[:,layer,:])
        c = numpy.mean(b, axis =0)
        activations.append(c)
        data = numpy.array(activations)
    return data

def synonyms():
    logging.getLogger().setLevel('INFO')
    with open("synonym_scores.txt", "w") as out:
        out.write("Representation Pair Error\n")
        #Activations per layer
        logging.info("Loading states")
        act = numpy.load('states.npy')


        i=0
        for k, synonympair in enumerate(split_into_pairs(act)):
            logging.info("Synonym pair {}".format(pairs[k]))
            for layer in range(5):
                labels= create_labels(synonympair)
                activations = layer_activation(synonympair, layer)
                predicted = cross_validation.cross_val_predict(LogisticRegression(), activations, labels, cv=10)
                error = 1 - accuracy_score(labels, predicted)
                out.write("rec{} {} {}\n".format(layer+1, pairs[k], error))
            i += 1


        # #MFCC
        logging.info("Loading MFCC")
        act = numpy.load('mfcc.npy')

        i=0
        for k, synonympair in enumerate(split_into_pairs(act)):
            logging.info("Synonym pair {}".format(pairs[k]))
            labels = create_labels(synonympair)
            data = timewise_average(synonympair)
            predicted = cross_validation.cross_val_predict(LogisticRegression(), data, labels, cv=10)
            error = 1 -  accuracy_score(labels,predicted)
            i += 1
            out.write("mfcc {} {}\n".format(pairs[k], error))

        logging.info("Loading Convo")
        act = numpy.load('conv_states.npy')

        i=0
        for k, synonympair in enumerate(split_into_pairs(act)):
            logging.info("Synonym pair {}".format(pairs[k]))
            labels = create_labels(synonympair)
            data = timewise_average(synonympair)
            predicted = cross_validation.cross_val_predict(LogisticRegression(), data, labels, cv=10)
            error = 1 -  accuracy_score(labels,predicted)
            i += 1
            out.write("conv {} {}\n".format(pairs[k], error))


        #Embeddings
        logging.info("Loading embeddings")
        act = numpy.load('embeddings.npy')
        i=0
        for k, synonympair in enumerate(split_into_pairs(act)):
            logging.info("Synonym pair {}".format(pairs[k]))
            labels = create_labels(synonympair)
            data = synonympair
            predicted = cross_validation.cross_val_predict(LogisticRegression(), data, labels, cv=10)
            error = 1 - accuracy_score(labels,predicted)
            i += 1
            out.write("emb {} {}\n".format(pairs[k], error))
