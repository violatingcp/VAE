import os,sys
os.environ['KERAS_BACKEND'] = 'tensorflow'

import numpy as np

import keras
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, Nadam, SGD
from keras.regularizers import l1
from keras.utils import np_utils
from keras.engine import Layer, InputSpec
from keras.layers import Input, Dense, Dropout, Activation, concatenate, BatchNormalization
from keras.models import Sequential,Model,model_from_json
from keras.callbacks import ModelCheckpoint
from keras.engine.topology import Layer
from keras.layers.merge import concatenate

def saveModel(model,odir,label):
    model_json = model.to_json()
    with open(odir+'/'+label+'.json', "w") as json_file:
        json_file.write(model_json)
    model.save_weights(odir+'/'+label+'.h5')

def loadModel(model,modeljson,modelweights):
    with open(modeljson,'r') as f:
        tmp_ = model_from_json(f.read())
    tmp_.load_weights(modelweights);
    model.set_weights(tmp_.get_weights())
    
def turnon(iD,iTrainable,iOther=0):
    """Turn on/off layers from model"""
    i0 = -1
    for l1 in iD.layers:
        i0=i0+1
        if iOther != 0 and l1 in iOther.layers:
            continue
        try:
            l1.trainable = iTrainable
        except:
            print("trainableErr",layer)

def partial_freeze(model, compargs):
    """Partial frozen model"""
    clone = Model(inputs=model.inputs, outputs=model.outputs)
    for l in clone.layers:
        if hasattr(l, 'freezable') and l.freezable:
            l.trainable = False 
    clone.compile(**compargs)
    return clone

def conditional_loss_function(y_true, y_pred):
    """Conditional loss function"""
    return categorical_crossentropy(y_true, y_pred)*(1-y_true[:,0])

def KL (y_true, y_pred):
    """
    Kullback-Leibler loss; maximises posterior p.d.f.
    """
    return -K.log(y_pred)
