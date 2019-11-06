#!/usr/bin/env python
import os,sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import keras
from keras import backend as K
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
import argparse 
# others
import numpy as np
import pandas as pd
import h5py
import matplotlib

# reader
import hinclusive.config as cfg
from hinclusive.reader._common import *

# train setup
from hinclusive.models._common import *

# setup
features_to_plot = ['j_pt','j_eta','j_mass_mmdt','j_rho']
features_range   = [(400.,1000.),(-2.4,2.4),(60.,160.),(-7,-1)]

#lFilePath = '../data/h5s_mass60to160_v10/ggh_and_qcd.h5'
lFilePath = '../data/h5s_mass60to160_v10/ggh_and_qcdweight.h5'
lFilePath_train= lFilePath.replace('.h5','_train.h5')
lFilePath_validate= lFilePath.replace('.h5','_validate.h5')
lFilePath_test= lFilePath.replace('.h5','_test.h5')
lFilePath_test = lFilePath_test.replace('weight','')

# columns declared in file
lColumns = getColumns(cfg)
print(len(lColumns))

nparts = cfg.nparts
lPartfeatures = []
for i0 in range(nparts):
    for iVar in cfg.lPartvars:
        lPartfeatures.append(iVar+str(i0))

# hids
hids = [0.,1.,2.,3.,4.,5.,6.]

# variables for training
labelh  = 'procid'
advhid  = 'h_decay_id1'
advmass = 'j_mass_mmdt'
advrho  = 'j_rho'

def ratio(var1,var2):
    return var1/var2
def getratio(df,var1,var2):
    x = np.vectorize(ratio)(df[var1],df[var2])
    return x

clfinputs = ['ratio_mmdt','ratio_trim','ratio_rsdb1','ratio_sdb1','ratio_prun','ratio_sdb2','ratio_sdm1']

def load(iFile):
    h5File = h5py.File(iFile)
    treeArray = h5File['test']
    df = pd.DataFrame(treeArray[()],columns=lColumns)
    h5File.close()
    df['ratio_mmdt'] = getratio(df,'j_mass','j_mass_mmdt')
    df['ratio_trim'] = getratio(df,'j_mass','j_mass_trim')
    df['ratio_rsdb1'] = getratio(df,'j_mass','j_mass_rsdb1')
    df['ratio_sdb1'] = getratio(df,'j_mass','j_mass_sdb1')
    df['ratio_prun'] = getratio(df,'j_mass','j_mass_prun')
    df['ratio_sdb2'] = getratio(df,'j_mass','j_mass_sdb2')
    df['ratio_sdm1'] = getratio(df,'j_mass','j_mass_sdm1')    
    features_val = df[clfinputs].values
    labels_val   = df[labelh].values
    advf_val     = df[advhid].values
    adv_val      = pd.get_dummies(advf_val) # one-hot encoding
    feat_val     = df[features_to_plot].values
    advmassf_val = df[advmass].values
    advmass_val  = xform_mass(df[advmass].values)
    advrho_val   = xform_rho(df[advrho].values)
    del h5File,df
    return features_val,labels_val,adv_val,advf_val,feat_val,advmassf_val,advmass_val,advrho_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--odir',        dest='odir',        default="ecfs",   help="odir")
    args = parser.parse_args()
    
    # load things
    X_train,Y_train,Y_adv_train,Y_advf_train,feat_train,mass_train,advmass_train,advrho_train = load(lFilePath_train)
    X_validate,Y_validate,Y_adv_validate,Y_advf_validate,feat_validate,mass_validate,advmass_validate,advrho_validate = load(lFilePath_validate)
    X_test,Y_test,Y_adv_test,Y_advf_test,feat_test,mass_test,advmass_test,advrho_test = load(lFilePath_test)

    import utils
    odir = args.odir
    os.system('mkdir -p %s'%odir)

    ranges = [(1, 2.5),(1, 2.5),(1, 2.5),(1, 2.5),(1, 2.5),(1, 2.5),(1, 2.5)]
    names = ['ratio_mmdt','ratio_trim','ratio_rsdb1','ratio_sdb1','ratio_prun','ratio_sdb2','ratio_sdm1']
    
    # plot inputs
    utils.plotSigvsBkg(Y_train,mass_train, xlabel="Jet mass [GeV]",label="msd",odir=odir)
    utils.plotInputs(Y_train,X_train,names,ranges,label='nonst_inputs',odir=odir)

    history = {}
    models = {}

    # calculate pt weights
    #ptweights = calc_ptweights(feat_train,Y_train)

    # standarize
    for x in [X_train,X_test,X_validate]:
        x -= x.mean(axis=0)
        x /= x.std (axis=0)
    num_vars  = len(clfinputs)
    from keras.layers import Input
    inputvars = Input(shape=X_train.shape[1:], name='input')
    input_mass = Input(shape=(1,), name='input_mass')

    # build classifier
    import hinclusive.models.ecfs as ecfs
    loss_classifier='binary_crossentropy'
    classifier = ecfs.build_classifier(inputvars,input_mass,num_vars,loss_classifier)
    models['classifier'] = classifier
    
    # run training
    print(models['classifier'])
    print('training ',Y_train)
    history['classifier'] = models['classifier'].fit(X_train,Y_train,
                                                     #batch_size=1000,epochs=50,verbose=1,
                                                     batch_size=500,epochs=200,verbose=1,
                                                     validation_data=[X_validate,Y_validate],
                                                     #sample_weight=ptweights
    )
            
    # get predictions
    y_pred_NN = models['classifier'].predict(X_test)

    # plot response and roc
    plotDecay = utils.plotByDecay('classifier',odir);
    plotDecay.test(models['classifier'],X_test,Y_test,Y_advf_test,feat_test)
    utils.plotOutputs(mass_test, Y_test,
                      [y_pred_NN, None] + X_test.T.tolist(),
                      ['NN classifier',None]+ names, label='outputs', odir=odir)
    # save classifier only
    saveModel(models['classifier'],odir,label='classifier')
