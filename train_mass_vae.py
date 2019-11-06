#!/usr/bin/env python
import os,sys
#os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import pandas as pd
import h5py

from keras.datasets import mnist
from keras.layers import Input,Dense,Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.optimizers import Adam, SGD

from sklearn.model_selection import train_test_split
#import matplotlib

#from hinclusive.models._common import *
#import hinclusive.config as cfg
#from hinclusive.reader._common import *

def extend_to(element, to=None):
    if to is None:
        to = []
    to.extend(element)
    return to

lPath = '/tmp/pharris'
lQFiles   = ['qcd700to1000.h5','qcd1000to1500.h5','qcd1500to2000.h5','qcd2000toInf.h5']
lHFiles   = ['ggh.h5']
lproc     = ['procid']
hids = [0.,1.,2.,3.,4.,5.,6.]
lfeatures = ['h_decay_id1','h_decay_id11','h_decay_id21','j_pt','j_eta','j_phi','j_mass','j_mass_mmdt','j_tau21_b1',
             'j_mass_trim','j_mass_rsdb1','j_mass_sdb1','j_mass_prun','j_mass_sdb2','j_mass_sdm1',
]
lecfs     = ['j_c1_b0_mmdt', 'j_c1_b1_mmdt', 'j_c1_b2_mmdt', 'j_c2_b1_mmdt', 'j_c2_b2_mmdt', 'j_d2_b1_mmdt', 'j_d2_b2_mmdt', 'j_d2_a1_b1_mmdt', 'j_d2_a1_b2_mmdt', 'j_m2_b1_mmdt', 'j_m2_b2_mmdt', 'j_n2_b1_mmdt', 'j_n2_b2_mmdt', 'j_m3_b1_mmdt', 'j_m3_b2_mmdt','j_n3_b1_mmdt', 'j_n3_b2_mmdt']
lmets     = ['met_pt','met_eta','met_phi','met_m','metsmear_pt','metsmear_phi']
lparts    = ['j_particles_pt','j_particles_phi','j_particles_eta','j_particles_id']
lnotintree = ['j_rho',
              'j_tau21ddt_b1','drmet_jet','drmetsmear_jet','met_px','met_py','met_pz',
              'metsmear_px','metsmear_py','metprojlepx','metprojlepy','metsmearprojlepx','metsmearprojlepy',
              'lep_pt','lep_eta','lep_phi','lep_m','drlep_jet','jetminuslep_mass',
              'mhgen','mhgen_mmdt','mhsmear','mhsmear_mmdt',
              'metjet_dphi','metsmearjet_dphi','metjet_deta'
]
lextras = []
extend_to(lecfs,lextras)
extend_to(lmets,lextras)
extend_to(lnotintree,lextras)
nparts = 40
lpartvars = ['j_part_pt_','j_part_phi_','j_part_eta_','j_part_id_']

# variables for training
labelh  = 'procid'
advhid  = 'h_decay_id1'
advmass = 'j_mass_mmdt'
advrho  = 'j_rho'
batch_size=10
z_dim=2

def getColumns():
    lcolumns = []
    lcolumns.extend(lproc)
    lcolumns.extend(lfeatures)
    lcolumns.extend(lextras)
    for i0 in range(nparts):
        for ivar in lpartvars:
                lcolumns.append(ivar+str(i0))
    return lcolumns

def ratio(var1,var2):
    return var1/var2

def getratio(df,var1,var2):
    x = np.vectorize(ratio)(df[var1],df[var2])
    return x

clfinputs = ['ratio_mmdt','ratio_trim','ratio_rsdb1','ratio_sdb1','ratio_prun','ratio_sdb2','ratio_sdm1']

def loaddata(iData):
    lColumns = getColumns()
    df = pd.DataFrame(iData,columns=lColumns)
    df['ratio_mmdt'] = getratio(df,'j_mass','j_mass_mmdt')
    df['ratio_trim'] = getratio(df,'j_mass','j_mass_trim')
    df['ratio_rsdb1'] = getratio(df,'j_mass','j_mass_rsdb1')
    df['ratio_sdb1'] = getratio(df,'j_mass','j_mass_sdb1')
    df['ratio_prun'] = getratio(df,'j_mass','j_mass_prun')
    df['ratio_sdb2'] = getratio(df,'j_mass','j_mass_sdb2')
    df['ratio_sdm1'] = getratio(df,'j_mass','j_mass_sdm1')    
    print len(df),"1"
    df.replace([np.inf, -np.inf], np.nan)
    df=df.dropna()
    print len(df),"2"
    features_val = df[clfinputs].values
    return features_val

def addh5(iFilePath,lFiles):
    tmpArray=[]
    i0 = 1
    for ifile in lFiles:
        lFile = iFilePath +'/' + ifile
        if not (os.path.isfile(lFile)): continue
        print(lFile)
        h5File = h5py.File(lFile)
 #       try:                                                                                                                                                                                                                                
            # tmp solution for ggh (because dataset got too big)                                                                                                                                                                             
        treeArray = h5File['test'][:200000]
        if 'ggh' in lFile:
            n = 20000#320262
            if(len(tmpArray)>n): continue
            print('ggh!, taking first %i elm instead of %i'%(n,len(h5File['test'][()])))
            treeArray = h5File['test'][:n]
        tmpArray.extend(treeArray)
#        except:                                                                                                                                                                                                                             
#            print('No evts')                                                                                                                                                                                                                
#            continue                                                                                                                                                                                                                        
        h5File.close()
        del h5File
        i0+=1
    print('total evts ',iFilePath,len(tmpArray))
    return tmpArray


def load(iPath,iFiles):
    data=addh5(iPath,iFiles)
    data_x=loaddata(data)
    return data_x

def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
    return mu + K.exp(log_var) * eps

if __name__ == "__main__":
    X_total  = load(lPath,lHFiles)
    #(X_total, y_tr),(x_te,y_te) = mnist.load_data()
    #X_total = X_total.astype('float32')/255.
    #X_total = X_total.reshape(X_total.shape[0], -1)
    print "!!!!",X_total
    X_train,X_test = train_test_split(X_total,test_size=0.4)
    # standarize
    for x in [X_train,X_test]:
        x -= x.mean(axis=0)
        x /= x.std (axis=0)
    num_vars  = len(clfinputs)
    x = Input(shape=X_train.shape[1:])
    print x.shape
    h = Dense(120, activation='relu')(x)
    h = Dense(80, activation='relu')(h)
    #h = Dense(5, activation='relu')(h)
    mu      = Dense(z_dim)(h)
    log_var = Dense(z_dim)(h)
    z = Lambda(sampling, output_shape=(z_dim,))([mu, log_var])
    z_decoder1 = Dense(20, activation='relu')
    z_decoder2 = Dense(80, activation='relu')
    z_decoder3 = Dense(120, activation='relu')
    y_decoder  = Dense(X_train.shape[1], activation='sigmoid')

    z_decoded = z_decoder1(z)
    z_decoded = z_decoder2(z_decoded)
    z_decoded = z_decoder3(z_decoded)
    y         = y_decoder (z_decoded)
    vae = Model(x,y)
    print "--->",X_train.shape[1],X_train.shape[1:]

    reconstruction_loss = objectives.binary_crossentropy(x, y) * X_train.shape[1]
    kl_loss = 0.5 * K.sum(K.square(mu) + K.exp(log_var) - log_var - 1, axis = -1)
    vae_loss = reconstruction_loss + kl_loss

    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam(lr=0.0001))#optimizer='rmsprop')#optimizer=Adam(lr=0.0001))
    vae.summary()
    vae.fit(X_train,
            #shuffle=True,
            batch_size=batch_size,epochs=100,
            validation_data=(X_test, None),verbose=1)
    
    model_json = vae.to_json()
    with open("vae_model.json", "w") as json_file:
        json_file.write(model_json)
    vae.save_weights("vae_model.h5")
    

    encoder = Model(x, mu)
    encoder.summary()
    encoder_json = encoder.to_json()
    with open("encoder_model.json", "w") as json_file:
        json_file.write(encoder_json)
    encoder.save_weights("encoder_model.h5")

    decoder_input = Input(shape=(z_dim,))
    _z_decoded = z_decoder1(decoder_input)
    _z_decoded = z_decoder2(_z_decoded)
    _z_decoded = z_decoder3(_z_decoded)
    _y = y_decoder(_z_decoded)
    generator = Model(decoder_input, _y)
    generator.summary()
    generator_json = generator.to_json()
    with open("generator_model.json", "w") as json_file:
        json_file.write(generator_json)
    generator.save_weights("generator_model.h5")
