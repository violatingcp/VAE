import os,sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#import tensorflow as tf
import keras
import numpy as np
#from keras import backend as K
#from tensorflow.python.client import device_lib
#device_lib.list_local_devices()
from optparse import OptionParser
import pandas as pd
import h5py
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam, Nadam, SGD
from keras.utils import to_categorical
#import matplotlib
#matplotlib.use('agg')
#%matplotlib inline
#import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Dropout, Activation, concatenate, BatchNormalization, GRU
from keras.models import Model 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc, roc_auc_score
import yaml

# reader
import hinclusive.config as cfg
from hinclusive.reader._common import *

# train setup
from hinclusive.models._common import *

fColors = {
'black'    : (0.000, 0.000, 0.000), # hex:000000
'blue'     : (0.122, 0.467, 0.706), # hex:1f77b4
'orange'   : (1.000, 0.498, 0.055), # hex:ff7f0e
'green'    : (0.173, 0.627, 0.173), # hex:2ca02c
'red'      : (0.839, 0.153, 0.157), # hex:d62728
'purple'   : (0.580, 0.404, 0.741), # hex:9467bd
'brown'    : (0.549, 0.337, 0.294), # hex:8c564b
'darkgrey' : (0.498, 0.498, 0.498), # hex:7f7f7f
'olive'    : (0.737, 0.741, 0.133), # hex:bcbd22
'cyan'     : (0.090, 0.745, 0.812)  # hex:17becf
}

colorlist = ['blue','orange','green','red','purple','brown','darkgrey','cyan']

features_to_plot = ['j_pt','j_mass_mmdt','j_n2_b1_mmdt']
features_range   = [(400.,1000.),(60.,160.),(0.,1.)]
nnout_cuts       = [0.1,0.2,0.4,0.6]

lFilePath = '/tmp/pharris/ggh_and_qcd.h5'
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

def turnon(iD,iTrainable,iOther=0):
    i0 = -1
    for l1 in iD.layers:
        i0=i0+1
        if iOther != 0 and l1 in iOther.layers:
            continue
        try:
            l1.trainable = iTrainable
        except:
            print("trainableErr",layer)

def load(iFile,iFileNpy='',iECFs=True,iNparts=40):
    h5File = h5py.File(iFile)
    treeArray = h5File['test'][()]

    labels='procid'
    adv   = 'h_decay_id1'

    features_labels_df = pd.DataFrame(treeArray,columns=lColumns)
    idconv = {11.:1, 12.:2, 13.:3, 22.:4, 130.:5, 211.:6, 310.:7, 321.:8, 2112.:9, 2212.:10, 3112.:11, 3122.:12, 3222.:13, 3312.:14, 3322.:15, 3334.:16, -11.:17, -12.:18, -13.:19, -22.:20, -130.:21, -211.:22, -310.:23, -321.:24, -2112.:25, -2212.:26, -3112.:27, -3122.:28, -3222.:29, -3312.:30, -3322.:31, -3334.:32, 0.:0}
    nIDs = 33
    for i0 in range(nparts):
        #print(abs(features_labels_df['j_part_id_0']).unique())
        features_labels_df['j_part_pt_'+str(i0)] = features_labels_df['j_part_pt_'+str(i0)]/features_labels_df['j_pt']
        features_labels_df['j_part_id_'+str(i0)] = features_labels_df['j_part_id_'+str(i0)].map(idconv)
    features_df        = features_labels_df[lPartfeatures]
    #features_df        = features_labels_df[lECFs_noNaNs]
    #features_val       = features_df.values
    labels_df          = features_labels_df[labels]
    adv_df             = features_labels_df[adv]
    labels_val         = labels_df.values # labels are procid either 0 or 1 
    advf_val           = adv_df.values  # this is hid:0,1,2...6
    adv_val            = pd.get_dummies(adv_df.values) # one-hot encoding     
    feat_val           = features_labels_df[features_to_plot].values
    print(features_labels_df['procid'].unique())
    print(features_labels_df['h_decay_id1'].unique())

    for p in lPartfeatures:
        if (features_df[p].isna().sum()>0): print(p,"found nan!!")

    #if iFileNpy!='':
    #    features_2df = np.load(iFileNpy)
    #else:
    features_2df = np.zeros((len(features_labels_df['procid']), nparts, len(cfg.lPartvars)+nIDs-1))
    for ir,row in features_df.iterrows():
        features_row =np.array(np.transpose(row.values.reshape(len(cfg.lPartvars),nparts)))
        features_row = np.concatenate((features_row[:,:-1],to_categorical(features_row[:,-1],num_classes=nIDs)),axis=1)
        features_2df[ir, :, :] = features_row
    features_val = features_2df 

    print(features_val)
    # split into random test and train subsets 
    X_train_val, X_test, y_train_val, y_test, y_adv_train, y_adv_test, y_advf_train, y_advf_test, feat_train, feat_test = train_test_split(features_val, labels_val, adv_val, advf_val, feat_val, test_size=0.2, random_state=42)
    #scaler = preprocessing.StandardScaler().fit(X_train_val)
    #X_train_val = scaler.transform(X_train_val)
    #X_test      = scaler.transform(X_test)
    return X_train_val, X_test, y_train_val, y_test, y_adv_train, y_adv_test, y_advf_train, y_advf_test, feat_train, feat_test

def conditional_loss_function(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred)*(1-y_true[:,0])

def model(Inputs,X_train,Y_train,Y_adv_train):
    NPARTS=20
    CLR=0.001
    LWR=0.001
    gru = GRU(100,activation='relu',recurrent_activation='hard_sigmoid',name='gru_base')(Inputs)
    dense   = Dense(100, activation='relu')(gru)
    norm    = BatchNormalization(momentum=0.6, name='dense4_bnorm')  (dense)
    dense   = Dense(50, activation='relu')(norm)
    norm    = BatchNormalization(momentum=0.6, name='dense5_bnorm')  (dense)
    dense   = Dense(20, activation='relu')(norm)
    dense   = Dense(10, activation='relu')(dense)
    out     = Dense(1, activation='sigmoid')(norm)
    classifier = Model(inputs=[Inputs], outputs=[out])
    lossfunction = 'binary_crossentropy'
    turnon(classifier,True) 
    classifier.compile(loss=[lossfunction], optimizer=Adam(CLR), metrics=['accuracy'])
    models={'classifier' : classifier}

    n_components = Y_adv_train.shape[1] # 7 decays bb WW tt cc glu glu ZZ other
    adv   = out
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(20, activation="relu")(adv)
    adv   = Dense(20, activation="relu")(adv)
    #adv   = Dense(n_components, activation="softmax")(adv)
    adv   = Dense(n_components, activation="sigmoid")(adv)
    #adv   = Dense(n_components, kernel_initializer='normal',name='adv') (adv)
    advM  = Model(inputs=[Inputs], outputs=[adv])
    models.update({'adv-base': advM})

    turnon(advM,False,classifier)
    turnon(classifier,True) 
    opt_DRf = SGD(momentum=0.0)
    DRf = Model(inputs=[Inputs], outputs=[classifier(Inputs), advM(Inputs)])
    DRf.compile(loss=[lossfunction,conditional_loss_function],loss_weights=[1.0,-1./LWR], optimizer=opt_DRf)
    models.update({'adv-full': DRf})

    turnon(classifier,False)
    turnon(advM,True,classifier)    
    opt_DfR = SGD(momentum=0.0)
    DfR = Model(inputs=[Inputs], outputs=[advM(Inputs)])
    models.update({'adv-back': DfR})
    turnon(classifier,False) 
    models['adv-back'].compile(loss=['categorical_crossentropy'], optimizer=opt_DfR)

    turnon(classifier,True) 
    classifier.compile(loss=[lossfunction], optimizer=Adam(CLR), metrics=['accuracy'])
    return models

def train(models,X_train,Y_train,Y_adv_train,feat_train):
    NEPOCHS=20
    Obatch_size=1000
    #Make signal only sample for adversary
    nbins=20
    ptbins = np.linspace(400.,1000.,num=nbins+1)
    sighist = np.zeros(nbins,dtype='f8')
    bkghist = np.zeros(nbins,dtype='f8')
    ptweights = np.ones(len(feat_train[:,0]),dtype='f8')
    for x in range(len(feat_train[:,0])):
        pti = 1
        while (pti<nbins):
            if (feat_train[x,0]>ptbins[pti-1] and feat_train[x,0]<ptbins[pti]): break
            pti = pti+1
        if (pti<nbins):
            if (Y_train[x]==1):
                sighist[pti] = sighist[pti]+1.
            else:
                bkghist[pti] = bkghist[pti]+1.
    sighist = sighist/sum(sighist)
    bkghist = bkghist/sum(bkghist)
    for y in range(len(sighist)):
        if (bkghist[y]>0.): print(sighist[y]/bkghist[y],)
        else: print(1.,)
    print('\n')
    for x in range(len(feat_train[:,0])):
        if (not Y_train[x]==0): continue
        pti = 1
        while (pti<nbins):
            if (feat_train[x,0]>ptbins[pti-1] and feat_train[x,0]<ptbins[pti]): break
            pti = pti+1
        if (pti<nbins and bkghist[pti]>0.): ptweights[x] = sighist[pti]/bkghist[pti]
    X_sig_train     = X_train    [Y_train==1]
    Y_sig_adv_train = Y_adv_train[Y_train==1]
    turnon(models['classifier'],True)
    #models['classifier'].fit(X_train, Y_train,    batch_size=1000,epochs=10,verbose=1)
    models['classifier'].fit(X_train, Y_train,    batch_size=500,epochs=15,verbose=1,sample_weight=ptweights)
    turnon(models['adv-base'],True,models['classifier'])
    turnon(models['classifier'],False)
    models['adv-back']  .fit(X_sig_train, Y_sig_adv_train,batch_size=500,epochs=3,verbose=1)
    for i in range(NEPOCHS):
        #l = models['adv-full'].evaluate(samples['X_val'], [samples['Y_cls_val'], samples['Y_adv_val']], verbose=0)
        #print(l[0],l[1],l[2])
        print('Starting epoch '+str(i+1)+'...')
        #opts['logfile'].write('\n\nTRAINING ADVERSARIAL NETWORK\n\n'+'\n')
        turnon(models['adv-base'],False,models['classifier'])
        turnon(models['classifier'],True)
        #indices = np.random.permutation(len(samples['X_train']))[:Obatch_size]
        #models['adv-full'].train_on_batch(samples['X_train'][indices], [samples['Y_cls_train'][indices], samples['Y_adv_train'][indices]])
        models['adv-full'].fit(X_train, [Y_train, Y_adv_train],epochs=1,verbose=1,batch_size=Obatch_size)
     
        turnon(models['classifier'],False)
        turnon(models['adv-base'],True,models['classifier'])
        #indices = np.random.permutation(len(samples['X_train']))[:Obatch_size]
        #models['adv-back'].train_on_batch(samples['X_train'][indices], samples['Y_adv_train'][indices])
        models['adv-back'].fit(X_sig_train,Y_sig_adv_train,epochs=1,verbose=1,batch_size=Obatch_size)
     
def plotNNResponse(data,labels):
    plt.clf()
    bins=20
    for j in range(len(data)):
      plt.hist(data[j],bins,log=False,histtype='step',normed=True,label=labels[j],fill=False,range=(0,1))
    plt.legend(loc='best')
    plt.xlabel('NeuralNet Response')
    plt.ylabel('Number of events (normalized)')
    plt.title('NeuralNet applied to test samples')
    plt.savefig("adversary_disc.pdf")
    plt.yscale('log')
    plt.savefig("adversary_disc_log.pdf")
    plt.yscale('linear')
    #plt.show()

def plotFeatResponse(data,feats,labels):

    feats_pass = []
    feats_fail = []
    for j in range(len(data)):
      bufpass = []
      buffail = []
      for c in nnout_cuts:
        tmppass = []
        tmpfail = []
        for x in range(len(data[j])):
          if (data[j][x]>c): tmppass.append(feats[j][x])
          else:              tmpfail.append(feats[j][x])
        bufpass.append(tmppass)
        buffail.append(tmpfail)
      feats_pass.append(bufpass)
      feats_fail.append(buffail)
    #[channel][cuts][entries][features]

    for fi in range(len(features_to_plot)):
      for ci in range(len(nnout_cuts)):
        plt.clf()
        bins=15

        for ip in range(2):
          for j in range(len(data)):
            temp = np.array([])
            pfstr = ""
            stylestr = ""
            doPlot = False
            if   (ip==0 and len(feats_pass[j][ci][:])>1): 
              temp = np.array(feats_pass[j][ci])
              temp = np.reshape(temp,(-1,len(features_to_plot)))
              pfstr = " pass"
              stylestr = "solid"
              if (len(temp[:,fi])>50): doPlot = True
            elif (ip==1 and len(feats_fail[j][ci][:])>1):
              temp = np.array(feats_fail[j][ci])
              temp = np.reshape(temp,(-1,len(features_to_plot)))
              pfstr = " fail"
              stylestr = "dashed"
              if (len(temp[:,fi])>50): doPlot = True
            #print(temp.shape)
            if (doPlot and j<4): plt.hist(temp[:,fi],bins,log=False,histtype='step',normed=True,linestyle=stylestr, label=labels[j]+pfstr,fill=False,range=features_range[fi],color=colorlist[j])
        plt.legend(loc='best')
        plt.xlabel(features_to_plot[fi])
        plt.ylabel('Number of events (normalized)')
        plt.title('NeuralNet < '+str(nnout_cuts[ci])+' applied to test samples')
        plt.savefig("adversary_"+features_to_plot[fi]+"_"+str(ci)+".pdf")
        plt.yscale('log')
        plt.savefig("adversary_"+features_to_plot[fi]+"_"+str(ci)+"_log.pdf")
        plt.yscale('linear')
        #plt.show()

def plotROC(truth, scores,labels):
    plt.clf()
    for j in range(len(truth)):
        x,y,_ =roc_curve(truth[j],scores[j])
        auc = roc_auc_score(truth[j],scores[j])
        plt.plot(x,y,label='{}, AUC = {:.2f}'.format(labels[j],auc))
    plt.legend(loc='lower right')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig("adversary_roc.pdf")
    #plt.show()

def test(models,X_test,Y_test,Y_adv_test,feat_test):
    model = models['classifier']
    subsamples = [ 
                  [0,0], #bkg
                  [1,1], #h bb
                  [1,2], #h WW
                  [1,3], #h tau tau 
                  [1,4], #h glu #glu
                  [1,5], #h ZZ
                  [1,6], #h cc
                  [1,0] ] #h other
    labels     = ["Bkg","hbb","hWW","htau tau ","h glu glu","h ZZ","h cc","h other"]
    roclabels  =       ["hbb","hWW","htau tau ","h glu glu","h ZZ","h cc","h other"]
    response_tests   = []
    response_preds   = []
    feat_preds       = []
    roc_preds        = []
    roc_true         = []
    i0=0
    print('in test')
    for subsample in subsamples:
        print(subsample)
        ids=np.logical_and(Y_test==subsample[0],Y_adv_test==subsample[1])
        tmpdata = X_test[ids]
        tmpfeat = feat_test[ids]
        tmppred = model.predict(tmpdata)
        print('\t',len(tmppred))
        response_tests.append(tmpdata)
        response_preds.append(tmppred)
        feat_preds.append(tmpfeat)
        if i0 > 0 and len(tmpdata) > 0:
            roc_true.append([])
            roc_true[-1].extend(np.zeros(len(response_tests[0] )))
            roc_true[-1].extend(np.ones (len(tmpdata)))
            roc_preds.append([])
            roc_preds[-1].extend(response_preds[0])
            roc_preds[-1].extend(tmppred)
        i0=i0+1

    print('now plotting resp')
    plotNNResponse(response_preds,labels)
    print('now plotting feats')
    plotFeatResponse(response_preds, feat_preds,labels)
    print('now plotting roc')
    plotROC(roc_true, roc_preds, roclabels) 


if __name__ == "__main__":
    X_train,X_test,Y_train,Y_test,Y_adv_train,Y_adv_test,Y_advf_train,Y_advf_test,feat_train,feat_test = load('/tmp/pharris/ggh_and_qcd.h5')
    inputvars=Input(shape=X_train.shape[1:], name='input')
    models = model(inputvars,X_train,Y_train,Y_adv_train)
    for m in models:
        print(str(m), models[m])
    train(models,X_train,Y_train,Y_adv_train,feat_train)
    print(len(Y_test),' vs ',sum(Y_test))
    test(models,X_test,Y_test,Y_advf_test,feat_test)
    for m in models:
        model_json = models[m].to_json()
        with open("model_"+str(m)+".json", "w") as json_file:
            json_file.write(model_json)
        models[m].save_weights("model_"+str(m)+".h5")
