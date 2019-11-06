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

lFilePath = '../data/h5s_mass60to160_v3/ggh_and_qcd.h5'
lFilePath_train= lFilePath.replace('.h5','_train.h5')
lFilePath_validate= lFilePath.replace('.h5','_validate.h5')
lFilePath_test= lFilePath.replace('.h5','_test.h5')

# columns declared in file
lColumns = getColumns(cfg)
nparts = cfg.nparts
lPartfeatures = []
for i0 in range(nparts):
    for iVar in cfg.lPartvars:
        lPartfeatures.append(iVar+str(i0))

lECFs_noNaNs = cfg.lECFs[:len(cfg.lECFs)-4]

# hids
hids = [0.,1.,2.,3.,4.,5.,6.]

# variables for training
labelh  = 'procid'
advhid  = 'h_decay_id1'
advmass = 'j_mass_mmdt'
advrho  = 'j_rho'

def load(iFile):
    h5File = h5py.File(iFile)
    treeArray = h5File['test']
    df = pd.DataFrame(treeArray[()],columns=lColumns)
    h5File.close()
    features_val = df[lECFs_noNaNs].values
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
    parser.add_argument('--trainclf',    dest='trainclf',    default=False,    help="train classifier", action='store_true')
    parser.add_argument('--trainmassadv',dest='trainmassadv',default=False,    help="train mass adv", action='store_true')
    parser.add_argument('--trainmassdecayadv',dest='trainmassdecayadv',default=False,    help="train mass and decay adv", action='store_true')
    parser.add_argument('--traindecayadv', dest='traindecayadv', default=False,    help="train decay adv", action='store_true')
    args = parser.parse_args()
    
    # load things
    X_train,Y_train,Y_adv_train,Y_advf_train,feat_train,mass_train,advmass_train,advrho_train = load(lFilePath_train)
    X_validate,Y_validate,Y_adv_validate,Y_advf_validate,feat_validate,mass_validate,advmass_validate,advrho_validate = load(lFilePath_validate)
    X_test,Y_test,Y_adv_test,Y_advf_test,feat_test,mass_test,advmass_test,advrho_test = load(lFilePath_test)

    import utils
    odir = args.odir
    os.system('mkdir -p %s'%odir)
    ranges = [(0, 0.6), (0, 0.4), (0, 0.4), (0, 0.4), (0, 0.4),
              (0, 4.0), (0, 4.0), (0, 4.0), (0, 4.0),
              (0, 0.5), (0, 0.5),
              (0, 1.0), (0, 1.0)]
    names  = lECFs_noNaNs   

    # plot inputs
    utils.plotSigvsBkg(Y_train,mass_train, xlabel="Jet mass [GeV]",label="msd",odir=odir)
    utils.plotInputs(Y_train,X_train,names,ranges,label='nonst_inputs',odir=odir)

    history = {}
    models = {}

    # calculate pt weights
    ptweights = calc_ptweights(feat_train,Y_train)

    # standarize
    for x in [X_train,X_test,X_validate]:
        x -= x.mean(axis=0)
        x /= x.std (axis=0)
    num_vars  = len(lECFs_noNaNs)
    from keras.layers import Input
    inputvars = Input(shape=X_train.shape[1:], name='input')
    input_mass = Input(shape=(1,), name='input_mass')

    # build classifier
    import hinclusive.models.ecfs as ecfs
    loss_classifier='binary_crossentropy'
    classifier = ecfs.build_classifier(inputvars,input_mass,num_vars,loss_classifier)
    models['classifier'] = classifier
    
    if args.trainclf:
        # run training
        history['classifier'] = models['classifier'].fit(X_train,Y_train,
                                                         batch_size=1000,epochs=10,verbose=1,
                                                         validation_data=[X_validate,Y_validate],
                                                         sample_weight=ptweights
        )
    else:
        loadModel(models['classifier'],'../metadata/models/ecfs-adversaries/model_classifier.json','../metadata/models/ecfs-adversaries/modelweights_classifier.h5')
            
    # get predictions
    y_pred_NN = models['classifier'].predict(X_test)

    if args.trainclf:
        # plot response and roc
        plotDecay = utils.plotByDecay('classifier',odir);
        plotDecay.test(models['classifier'],X_test,Y_test,Y_advf_test,feat_test)
        utils.plotOutputs(mass_test, Y_test,
                          [y_pred_NN, None] + X_test.T.tolist(),
                          ['NN classifier',None]+ names, label='outputs', odir=odir)
        # save classifier only
        saveModel(models['classifier'],odir,label='classifier')

    if args.trainmassadv:
        # build mass adversary
        lam = 10.                     # Regularisation parameter, lambda
        loss_weights = [1.0E-08, 1.]  # Relative learning rates for classifier and adversary, resp.
        lr_ratio = loss_weights[0] / loss_weights[1]
        
        # Construct mass adversary model
        models['adversary'] = ecfs.adversary_model(num_vars)
        
        # Construct combined model
        models['combined'] = ecfs.combined_model(models['classifier'], models['adversary'], lam, lr_ratio,loss_classifier,loss_weights)
        
        # Prepare sample weights (i.e. only do mass-decorrelation for backround)
        sample_weight = [np.ones(int(X_train.shape[0]), dtype=float), (Y_train == 0).astype(float)]
        sample_weight[1] *= np.sum(sample_weight[0]) / np.sum(sample_weight[1])
        
        # Rescale jet mass to [0,1]
        mt_train  = mass_train - mass_train.min()
        mt_train /= mt_train.max()
        mt_validate  = mass_validate - mass_validate.min()
        mt_validate /= mt_validate.max()
        
        # Get classifier predictions
        z_train = models['classifier'].predict(X_train)
        z_validate = models['classifier'].predict(X_validate)
        utils.plotPosterior(models['adversary'], mass_train[Y_train == 0], z_train[Y_train == 0], title='Before adversarial training', label='before_adversary_mass',odir=odir);

        # fit adversary only
        history['adversary'] = models['adversary'].fit([z_train.flatten(), mt_train],
                                                       np.ones_like(mt_train),
                                                       sample_weight=sample_weight[1],
                                                       batch_size=5000,
                                                       epochs=5,
                                                       validation_data=[[z_validate.flatten(),mt_validate],
                                                                        np.ones_like(mt_validate)]
        )
        '''
        # fit combined
        history['combined'] = models['combined'].fit([X_train, mt_train],
                                                     [Y_train, np.ones_like(mt_train)],
                                                     sample_weight=sample_weight, epochs=100,
                                                     batch_size=1000,
                                                     validation_data =[[X_validate, mt_validate],
                                                                       [Y_validate, np.ones_like(mt_validate)]])

        # save model
        saveModel(models['classifier'],odir,label='adversary_mass')
        
        # adversary predictions
        z_train    = models['classifier'].predict(X_train)
        y_pred_ANN = models['classifier'].predict(X_test)
        print('z_train mass adv ',len(z_train),z_train,z_train.shape,z_train[0].shape)
        print('y_predAN mass adv ',len(y_pred_ANN),y_pred_ANN,y_pred_ANN.shape)
        
        utils.plotPosterior(models['adversary'], mass_train[Y_train == 0], z_train[Y_train == 0], title='After adversarial training', label='after_adversary_mass',odir=odir);

        # plot ROC again
        plotAdv = utils.plotByDecay('adversary_mass',odir);
        plotAdv.test(models['classifier'],X_test,Y_test,Y_advf_test,feat_test)
        utils.plotOutputs(mass_test, Y_test,
                          [y_pred_ANN, y_pred_NN] + X_test.T.tolist(),
                          ['ANN classifier','NN classifier']+ names, label='outputs_adversary_mass', odir=odir)
        '''
    if args.trainmassdecayadv:
        # Construct decay adversary model
        lam = 10.                     # Regularisation parameter, lambda
        loss_weights = [1.0E-08, 1., 10.]  # Relative learning rates for classifier and adversary, resp.
        #LWR = 0.001
        #loss_weights = [1.0,1./LWR,1./LWR]
        lr_ratio = loss_weights[0] / loss_weights[1]
        models['adversary_mass_decay'] = ecfs.adversary_mass_decay(num_vars,Y_adv_train,[1.,1.])
        
        # Construct combined model
        models['combined_mass_decay'] = ecfs.combined_model_mass_decay(models['classifier'], models['adversary_mass_decay'],
                                                                       lam, lr_ratio,loss_classifier,loss_weights)
        
        # prepare sample weights
        sample_weight = [np.ones(int(X_train.shape[0]), dtype=float), (Y_train == 0).astype(float), (Y_train == 1).astype(float)]
        print('sample weight ',sample_weight)
        print('sum 0 ',np.sum(sample_weight[0]))
        print('sum 1 ',np.sum(sample_weight[1]))
        print('sum 2 ',np.sum(sample_weight[2]))
        sample_weight[1] *= np.sum(sample_weight[0]) / np.sum(sample_weight[1])
        sample_weight[2] *= np.sum(sample_weight[0]) / np.sum(sample_weight[2])
        #sample_weight[2] *= 1 / np.sum(sample_weight[2])
        print('sample weight 1 ',sample_weight[1])
        print('sample weight 2 ',sample_weight[2])
        print('sample weight ',sample_weight)
                
        # Rescale jet mass to [0,1]
        mt_train  = mass_train - mass_train.min()
        mt_train /= mt_train.max()
        mt_validate  = mass_validate - mass_validate.min()
        mt_validate /= mt_validate.max()
        
        # get classifier pred
        z_train = models['classifier'].predict(X_train)
        z_validate = models['classifier'].predict(X_validate)
        
        # fit adversary only
        # feed bkg only for mass adv
        z_train_bkg = models['classifier'].predict(X_train[Y_train==0])
        mt_train_bkg = mt_train[Y_train==0]
        z_validate_bkg = models['classifier'].predict(X_validate[Y_validate==0])
        mt_validate_bkg = mt_validate[Y_validate==0]
        
        # and sig only for decay adv
        z_train_sig = models['classifier'].predict(X_train[Y_train==1])
        Y_adv_train_sig = Y_adv_train[Y_train==1]
        z_validate_sig = models['classifier'].predict(X_validate[Y_validate==1])
        Y_adv_validate_sig = Y_adv_validate[Y_validate==1]
        
        history['adversary_mass_decay'] = models['adversary_mass_decay'].fit(#[z_train.flatten(), mt_train,Y_adv_train],
            #[z_train_bkg.flatten(), z_train_sig.flatten(), mt_train_bkg],
            [z_train.flatten(), mt_train],
            [np.ones_like(mt_train), Y_adv_train],
            batch_size=1000,
            epochs=5,
            #sample_weight=[sample_weight[1],sample_weight[2]],
            verbose=1,
            validation_data=[#[z_validate.flatten(),mt_validate,Y_adv_validate],
                [z_validate.flatten(), mt_validate],
                [np.ones_like(mt_validate), Y_adv_validate]])

        utils.plotPosterior(models['adversary_mass_decay'],
                            mass_train[Y_train == 0], z_train[Y_train == 0], title='After adversarial training', label='after_adversary_mass_decay',odir=odir,ipred=0);
        
        # fit combined
        history['combined_mass_decay'] = models['combined_mass_decay'].fit([X_train, mt_train],
                                                                           [Y_train, np.ones_like(mt_train), Y_adv_train],
                                                                           #sample_weight=sample_weight,
                                                                           epochs=100,
                                                                           batch_size=1000,
                                                                           validation_data =[[X_validate, mt_validate],
                                                                                             [Y_validate, np.ones_like(mt_validate), Y_adv_validate]]) 
        
        # save model
        saveModel(models['classifier'],odir,label='adversary_mass_decay')
        saveModel(models['adversary_mass_decay'],odir,label='adversary_mass_decay_only')

        # adv predictions
        z_train    = models['classifier'].predict(X_train)
        y_pred_ANN = models['classifier'].predict(X_test)
        print('z_train mass adv ',len(z_train),z_train,z_train.shape,z_train[0].shape)
        print('y_predAN mass adv ',len(y_pred_ANN),y_pred_ANN,y_pred_ANN.shape)

        # plot ROC again
        plotAdv = utils.plotByDecay('adversary_mass_decay',odir);
        plotAdv.test(models['classifier'],X_test,Y_test,Y_advf_test,feat_test)
        utils.plotOutputs(mass_test, Y_test,
                          [y_pred_ANN, y_pred_NN] + X_test.T.tolist(),
                          ['ANN classifier','NN classifier']+ names, label='outputs_adversary_mass_decay', odir=odir)  

    if args.traindecayadv:
        sample_weight = [np.ones(int(X_train.shape[0]), dtype=float), (Y_train == 1).astype(float)]
        sample_weight[1] *= np.sum(sample_weight[0]) / np.sum(sample_weight[1])
    
        z_train = models['classifier'].predict(X_train)
        z_validate = models['classifier'].predict(X_validate)
        '''
        models['adversary_decay_base'],models['adversary_decay_combined'],models['adversary_decay_back'] = ecfs.build_adversary_decay(Y_adv_train, models['classifier'],loss_classifier)
        history['adversary_decay_back'] = models['adversary_decay_back'].fit(X_train,
                                                                             Y_adv_train,
                                                                         sample_weight=sample_weight[1],
                                                                             epochs=5,
                                                                             batch_size=1000,
                                                                             validation_data=[X_validate,Y_adv_validate])
        
        utils.plotPosterior(models['adversary_decay_back'],
                            Y_adv_train[Y_train == 1], z_train[Y_train == 1], title='After adversarial training', label='decay1_after_adversary_mass_decay',odir=odir,ipred=1);

        history['adversary_decay_combined'] = models['adversary_decay_combined'].fit(X_train,
                                                                                     [Y_train, Y_adv_train],
                                                                                     sample_weight=sample_weight,
                                                                                     epochs=10,
                                                                                     batch_size=1000,
                                                                                     validation_data=[X_validate,
                                                                                                      [Y_validate,Y_adv_validate]])
        

        '''
        lam = 10.                     # Regularisation parameter, lambda
        #loss_weights = [1.0E-08, 1.]  # Relative learning rates for classifier and adversary, resp.
        LWR = 0.001
        loss_weights = [1.0,1./LWR]
        lr_ratio = loss_weights[0] / loss_weights[1]
        # construct adv only
        models['adversary_decay_only'] = ecfs.adversary_decay_only(Y_adv_train,models['classifier'])
        # Construct combined model
        models['combined_decay_only'] = ecfs.combined_model_decay(models['classifier'],
                                                                  models['adversary_decay_only'],
                                                                  lam, lr_ratio,loss_classifier,loss_weights)

        z_train_sig = models['classifier'].predict(X_train[Y_train==1])
        z_validate_sig =  models['classifier'].predict(X_validate[Y_validate==1])
        Y_adv_train_sig = Y_adv_train[Y_train==1]
        Y_adv_validate_sig = Y_adv_validate[Y_validate==1]
        
        models['adversary_decay_only'].fit(##[z_train_sig.flatten()],
                                           #X_train,#[np.ones_like(Y_adv_train)], # this was uncom
                                           ##Y_adv_train_sig, #this was uncom
                                           [z_train.flatten()],
                                           Y_adv_train,
                                           sample_weight=sample_weight[1], # this was comm
                                           batch_size=500, epochs=3,
                                           #validation_data=[z_validate_sig.flatten(),Y_adv_validate_sig])
                                           validation_data=[z_validate.flatten(),Y_adv_validate])
                                           
        models['combined_decay_only'].fit(X_train,
                                          [Y_train, Y_adv_train],
                                          #sample_weight=sample_weight,
                                          epochs=30,
                                          batch_size=1000,
                                          validation_data =[X_validate,[Y_validate, Y_adv_validate]])

        # plot ROC again
        plotAdv = utils.plotByDecay('adversary_decay',odir);
        plotAdv.test(models['classifier'],X_test,Y_test,Y_advf_test,feat_test)
        #saveModel(models['classifier'],odir,label='adversary_decay_lwr0001')
        
