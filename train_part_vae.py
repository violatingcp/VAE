#!/usr/bin/env python
import os,sys
#os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import pandas as pd
import h5py

from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.layers import Input,Dense,Lambda,GRU,LSTM,BatchNormalization,RepeatVector,TimeDistributed
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical

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

lFilePath = '/tmp/pharris/ggh.h5'
lPath = '/tmp/pharris'
lQFiles   = ['qcd700to1000.h5','qcd1000to1500.h5','qcd1500to2000.h5','qcd2000toInf.h5']
lHFiles   = ['ggh.h5']

lProc     = ['procid']
hids = [0.,1.,2.,3.,4.,5.,6.]
lFeatures = ['h_decay_id1','h_decay_id11','h_decay_id21','j_pt','j_eta','j_phi','j_mass','j_mass_mmdt','j_tau21_b1',
             'j_mass_trim','j_mass_rsdb1','j_mass_sdb1','j_mass_prun','j_mass_sdb2','j_mass_sdm1',
]
lECFs     = ['j_c1_b0_mmdt', 'j_c1_b1_mmdt', 'j_c1_b2_mmdt', 'j_c2_b1_mmdt', 'j_c2_b2_mmdt', 'j_d2_b1_mmdt', 'j_d2_b2_mmdt', 'j_d2_a1_b1_mmdt', 'j_d2_a1_b2_mmdt', 'j_m2_b1_mmdt', 'j_m2_b2_mmdt', 'j_n2_b1_mmdt', 'j_n2_b2_mmdt', 'j_m3_b1_mmdt', 'j_m3_b2_mmdt','j_n3_b1_mmdt', 'j_n3_b2_mmdt']
lMets     = ['met_pt','met_eta','met_phi','met_m','metsmear_pt','metsmear_phi']
lParts    = ['j_particles_pt','j_particles_phi','j_particles_eta','j_particles_id']
lNotInTree = ['j_rho',
              'j_tau21ddt_b1','dRmet_jet','dRmetsmear_jet','met_px','met_py','met_pz',
              'metsmear_px','metsmear_py','metprojlepx','metprojlepy','metsmearprojlepx','metsmearprojlepy',
              'lep_pt','lep_eta','lep_phi','lep_m','dRlep_jet','jetminuslep_mass',
              'mhgen','mhgen_mmdt','mhsmear','mhsmear_mmdt',
              'metjet_dPhi','metsmearjet_dPhi','metjet_dEta'
]
lExtras = []
extend_to(lECFs,lExtras)
extend_to(lMets,lExtras)
extend_to(lNotInTree,lExtras)
lPartvars = ['j_part_pt_','j_part_phi_','j_part_eta_','j_part_id_']
nparts = 40
lPartfeatures = []
for i0 in range(nparts):
    for iVar in lPartvars:
        lPartfeatures.append(iVar+str(i0))

# variables for training
labelh  = 'procid'
advhid  = 'h_decay_id1'
advmass = 'j_mass_mmdt'
advrho  = 'j_rho'
batch_size=500
z_dim=2

def getColumns():
    lColumns = []
    lColumns.extend(lProc)
    lColumns.extend(lFeatures)
    lColumns.extend(lExtras)
    for i0 in range(nparts):
        for iVar in lPartvars:
                lColumns.append(iVar+str(i0))
    print len(lColumns)
    return lColumns

def ratio(var1,var2):
    return var1/var2

def getratio(df,var1,var2):
    x = np.vectorize(ratio)(df[var1],df[var2])
    return x

clfinputs = ['ratio_mmdt','ratio_trim','ratio_rsdb1','ratio_sdb1','ratio_prun','ratio_sdb2','ratio_sdm1']

def loadData(iData):
    lColumns = getColumns()
    #h5File = h5py.File(iFile)
    #treeArray = h5File['test'][()]
    df = pd.DataFrame(iData,columns=lColumns)
    print "A"
    idconv = {11.:1, 12.:2, 13.:3, 22.:4, 130.:5, 211.:6, 310.:7, 321.:8, 2112.:9, 2212.:10, 3112.:11, 3122.:12, 3222.:13, 3312.:14, 3322.:15, 3334.:16, -11.:17, -12.:18, -13.:19, -22.:20, -130.:21, -211.:22, -310.:23, -321.:24, -2112.:25, -2212.:26, -3112.:27, -3122.:28, -3222.:29, -3312.:30, -3322.:31, -3334.:32, 0.:0}
    nIDs = 33
    print "B"
    for i0 in range(nparts):
        df['j_part_pt_'+str(i0)] = df['j_part_pt_'+str(i0)]/df['j_pt']
        df['j_part_id_'+str(i0)] = df['j_part_id_'+str(i0)].map(idconv)
    features_val = df[lPartfeatures]
    for p in lPartfeatures:
        if (df[p].isna().sum()>0): print(p,"found nan!!")

    features_2df = np.zeros((len(df['procid']), nparts, len(lPartvars)+nIDs-1))
    for ir,row in features_val.iterrows():
        features_row =np.array(np.transpose(row.values.reshape(len(lPartvars),nparts)))
        features_row = np.concatenate((features_row[:,:-1],to_categorical(features_row[:,-1],num_classes=nIDs)),axis=1)
        features_2df[ir, :, :] = features_row
    features_val = features_2df 
    return features_val

def addh5(iFilePath,lFiles):
    tmpArray=[]
    i0 = 1
    for ifile in lFiles:
        lFile = iFilePath +'/' + ifile
        if not (os.path.isfile(lFile)): continue
        print(lFile)
        h5File = h5py.File(lFile)
        treeArray = h5File['test'][0:150000]
        #if 'ggh' in lFile:
        #    n = 20000#320262
        #    if(len(tmpArray)>n): continue
        #    print('ggh!, taking first %i elm instead of %i'%(n,len(h5File['test'][()])))
        #    treeArray = h5File['test'][:n]
        tmpArray.extend(treeArray)
        h5File.close()
        del h5File
        i0+=1
    print('total evts ',iFilePath,len(tmpArray))
    return tmpArray


def load(iPath,iFiles):
    data=addh5(iPath,iFiles)
    data_x=loadData(data)
    return data_x

def sampling(args):
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
    return mu + K.exp(log_var) * eps


if __name__ == "__main__":
    X_total  = load(lPath,lQFiles)
    #X_total  = load(lFilePath)
    #(X_total, y_tr),(x_te,y_te) = mnist.load_data()
    X_train,X_test = train_test_split(X_total,test_size=0.2)
    print X_train.shape,X_test.shape,X_test[0]
    # standarize
    #for x in [X_train,X_test]:
    #    x -= x.mean(axis=0)
    #    x /= x.std (axis=0)
    num_vars  = len(clfinputs)
    x = Input(shape=X_train.shape[1:])
    h = GRU(100,activation='relu',recurrent_activation='hard_sigmoid',name='gru_base')(x)
    h = Dense(50, activation='relu')(h)
    h = BatchNormalization(momentum=0.6, name='dense4_bnorm')(h)
    h = Dense(20, activation='relu')(h)
    h = BatchNormalization(momentum=0.6, name='dense5_bnorm')(h)
    h = Dense(10, activation='relu')(h)
    mu      = Dense(z_dim)(h)
    log_var = Dense(z_dim)(h)
    z = Lambda(sampling, output_shape=(z_dim,))([mu, log_var])
    z_decoder1 = Dense(10, activation='relu')
    z_decoder2 = BatchNormalization(momentum=0.6, name='dense6_bnorm')
    z_decoder3 = Dense(20, activation='relu')
    z_decoder4 = BatchNormalization(momentum=0.6, name='dense7_bnorm')
    z_decoder5 = Dense(50, activation='relu')
    z_decoder6 = RepeatVector(nparts)
    z_decoder7 = GRU(100,return_sequences=True)
    z_decoder8 = TimeDistributed(Dense(36, activation='softmax'), name='decoded_mean')

    z_decoded = z_decoder1(z)
    z_decoded = z_decoder2(z_decoded)
    z_decoded = z_decoder3(z_decoded)
    z_decoded = z_decoder4(z_decoded)
    z_decoded = z_decoder5(z_decoded)
    z_decoded = z_decoder6(z_decoded)
    z_decoded = z_decoder7(z_decoded)
    y         = z_decoder8(z_decoded)
    vae = Model(x,y)
    print "--->",X_train.shape[0:],y.shape,x.shape,"----->",X_train.shape[1] 

    #def vae_loss(x, x_decoded_mean):
    x1 = K.flatten(x)
    y1 = K.flatten(y)
    print "shape",x.shape,x1.shape,"-",y.shape,y1.shape
    xent_loss = nparts * objectives.binary_crossentropy(x1,y1)
    kl_loss = - 0.5 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
    vae_loss = xent_loss + kl_loss

    vae.add_loss(vae_loss)
    vae.compile(optimizer='Adam')#,loss=[vae_loss],metrics=['accuracy'])
    vae.summary()
    vae.fit(X_train,
            shuffle=True,
            batch_size=batch_size,epochs=30,
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
    _z_decoded = z_decoder4(_z_decoded)
    _z_decoded = z_decoder5(_z_decoded)
    _z_decoded = z_decoder6(_z_decoded)
    _z_decoded = z_decoder7(_z_decoded)
    _y = z_decoder8(_z_decoded)
    generator = Model(decoder_input, _y)
    generator.summary()
    generator_json = generator.to_json()
    with open("generator_model.json", "w") as json_file:
        json_file.write(generator_json)
    generator.save_weights("generator_model.h5")
