import numpy as np
from keras.utils import np_utils

""" Shuffle """
def chunks(lColumns,path, chunksize, max_q_size=4, shuffle=True):
    """Yield successive n-sized chunks from a and b."""
    import h5py
    h5File = h5py.File(path)
    tree = h5File['test'][()]
    df = pd.DataFrame(treeArray,columns=lColumns)
    for istart in range(0,nrows,max_q_size*chunksize):
        a = preprocess_inputs(df[partfeatures][istart:istart+max_q_size*chunksize]) # Features
        b = df[labels][istart:istart+max_q_size*chunksize].values() # Labels
        if shuffle:
            c = np.c_[a.reshape(len(a), -1), b.reshape(len(b), -1)] # shuffle within queue size
            np.random.shuffle(c)
            test_features = c[:, :a.size//len(a)].reshape(a.shape)
            test_labels = c[:, a.size//len(a):].reshape(b.shape)
        else:
            test_features = a
            test_labels = b
        for jstart in range(0,len(test_labels),chunksize):
            yield test_features[jstart:jstart+chunksize].copy(),test_labels[jstart:jstart+chunksize].copy(), len(test_labels[jstart:jstart+chunksize].copy())

def preprocess_inputs(df):
    """Preprocess particles features df to values"""
    import numpy as np
    for i0 in range(nparts):
        df['j_part_pt_'+str(i0)] = df['j_part_pt_'+str(i0)]/df['j_pt']
        df['j_part_id_'+str(i0)] = df['j_part_id_'+str(i0)].map(idconv)
    features_2df = np.zeros((len(df['procid']), nparts, len(partvars)+nIDs-1))
    for ir,row in features_df.iterrows():
        features_row = np.array(np.transpose(row.values.reshape(len(partvars),nparts)))
        features_row = np.concatenate((features_row[:,:-1],to_categorical(features_row[:,-1],num_classes=nIDs)),axis=1)
        features_2df[ir, :, :] = features_row
        features_val = features_2df
    return features_val

def convert_to_testdf(df,columns):
    """Return sorted df"""
    df2 = df[columns]
    return df2

def xform_mass(x):
    n_decorr_bins = 15
    max_mass = 160.
    msd_norm_factor = 1. / max_mass
    binned = (np.minimum(x, max_mass) * msd_norm_factor * (n_decorr_bins - 1)).astype(np.int)
    onehot = np_utils.to_categorical(binned, n_decorr_bins)
    return onehot

def xform_rho(x):
    n_decorr_bins = 15
    max_rho = -1.
    min_rho = -7.
    rho_norm_factor = 1./(max_rho-min_rho)
    #binned = (abs(np.minimum(np.maximum(x, min_rho),max_rho)) * rho_norm_factor * (n_decorr_bins - 1)).astype(np.int)
    binned = ((np.minimum(np.maximum(x, min_rho),max_rho)-min_rho) * rho_norm_factor * (n_decorr_bins - 1)).astype(np.int)
    onehot = np_utils.to_categorical(binned, n_decorr_bins)
    return onehot

def calc_ptweights(feat_train,Y_train):
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
    for x in range(len(feat_train[:,0])):
        if (not Y_train[x]==0): continue
        pti = 1
        while (pti<nbins):
            if (feat_train[x,0]>ptbins[pti-1] and feat_train[x,0]<ptbins[pti]): break
            pti = pti+1
        if (pti<nbins and bkghist[pti]>0.): ptweights[x] = sighist[pti]/bkghist[pti]
    return ptweights

"""To concatenate h5s"""
def addh5(iFilePath,lFiles):
    tmpArray=[]
    i0 = 1
    for ifile in lFiles:
        lFile = iFilePath +'/' + ifile
        if not (os.path.isfile(lFile)): continue
        h5File = h5py.File(lFile)
        try:
            treeArray = h5File['test'][()]
            tmpArray.extend(treeArray)
        except:
            print('No evts')
            continue
        h5File.close()
        del h5File
        i0+=1
    return tmpArray

""" give columns"""
def getColumns(cfg):
    lColumns = []
    lColumns.extend(cfg.lProc)
    lColumns.extend(cfg.lFeatures)
    lColumns.extend(cfg.lExtras)
    for i0 in range(cfg.nparts):
        for iVar in cfg.lPartvars:
            lColumns.append(iVar+str(i0))
    return lColumns
