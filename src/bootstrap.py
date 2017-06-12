import numpy
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import logging

def error_rate(data):
    return 1-numpy.array([accuracy_score(data[:,0], data[:,i]) for i in range(1, data.shape[1])])

def bootstrap():
    logging.info("Loading results for bootstrap")
    data_mfcc = numpy.load("lr_mfcc_pred.npy")
    data_conv = numpy.load("lr_conv_pred.npy")
    data_rec0 = numpy.load("lr_recurrent0_pred.npy")
    data_rec1 = numpy.load("lr_recurrent1_pred.npy")
    data_rec2 = numpy.load("lr_recurrent2_pred.npy")
    data_rec3 = numpy.load("lr_recurrent3_pred.npy")
    data_rec4 = numpy.load("lr_recurrent4_pred.npy")
    data = numpy.vstack([ data_mfcc[:,0],
                          data_mfcc[:,1],
                          data_conv[:,1],
                          data_rec0[:,1],
                          data_rec1[:,1],
                          data_rec2[:,1],
                          data_rec3[:,1],
                          data_rec4[:,1]
                          ]).T
    logging.info("Resampling")
    err_boot = numpy.array([ error_rate(resample(data)) for _ in range(1000) ])
    logging.info("Saving bootstrapped results to err_boot.csv")
    numpy.savetxt("err_boot.csv",err_boot)
