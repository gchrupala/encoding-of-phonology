
# coding: utf-8

# In[2]:

get_ipython().magic(u'pylab inline --no-import-all')


# In[4]:

import numpy
from sklearn.utils import resample
from sklearn.metrics import accuracy_score


# In[23]:

def error_rate(data):
    return 1-numpy.array([accuracy_score(data[:,0], data[:,i]) for i in range(1, data.shape[1])])


# In[68]:

data_mfcc = numpy.load("/home/gchrupala/phonemes/data/lr_mfcc_pred.npy")
data_conv = numpy.load("/home/gchrupala/phonemes/data/lr_conv_pred.npy")
data_rec0 = numpy.load("/home/gchrupala/phonemes/data/lr_recurrent0_pred.npy")
data_rec1 = numpy.load("/home/gchrupala/phonemes/data/lr_recurrent1_pred.npy")
data_rec2 = numpy.load("/home/gchrupala/phonemes/data/lr_recurrent2_pred.npy")
data_rec3 = numpy.load("/home/gchrupala/phonemes/data/lr_recurrent3_pred.npy")
data_rec4 = numpy.load("/home/gchrupala/phonemes/data/lr_recurrent4_pred.npy")

data = numpy.vstack([ data_mfcc[:,0], 
                      data_mfcc[:,1],   
                      data_conv[:,1],
                      data_rec0[:,1],
                      data_rec1[:,1],
                      data_rec2[:,1],
                      data_rec3[:,1],
                      data_rec4[:,1]
                      ]).T


# In[69]:

err_boot = numpy.array([ error_rate(resample(data)) for _ in range(1000) ])


# In[70]:

numpy.savetxt("err_boot.csv",err_boot)


# In[61]:

plt.figure(figsize=(10,7))
plt.boxplot(err_boot[:,:], labels=["MFCC", "Conv", "Rec1", "Rec2","Rec3"], vert=False)
plt.xlabel("Error rate")


# In[51]:

get_ipython().magic(u'pinfo plt.boxplot')


# In[ ]:



