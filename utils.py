#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

class plotByDecay:
    def __init__(self, key = 'classifier', odir ='./',
                 features_to_plot = ['j_pt','j_eta','j_mass_mmdt','j_rho'],
                 features_range   = [(400.,1000.),(-2.4,2.4),(60.,160.),(-7,-1)]):
        self._features_to_plot = features_to_plot
        self._features_range = features_range
        self._colorlist = ['blue','orange','green','red','purple','brown','darkgrey','cyan']
        self._nnout_cuts       = [0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0]
        self._key = key
        self._odir = odir
        
    def plotNNResponse(self,data,labels):
        plt.clf()
        bins=20
        for j in range(len(data)):
            plt.hist(data[j],bins,log=False,histtype='step',normed=True,label=labels[j],fill=False,range=(0,1))
        plt.legend(loc='best')
        plt.xlabel('NeuralNet Response')
        plt.ylabel('Number of events (normalized)')
        plt.title('NeuralNet applied to test samples')
        plt.savefig(self._odir+"/"+self._key+"_disc.pdf")
        plt.yscale('log')
        plt.savefig(self._odir+"/"+self._key+"_disc_log.pdf")
        plt.yscale('linear')
        plt.show()

    def plotFeatResponse(self,data,feats,labels):
        feats_pass = []
        feats_fail = []
        for j in range(len(data)):
            bufpass = []
            buffail = []
            for c in self._nnout_cuts:
                tmppass = []
                tmpfail = []
                for x in range(len(data[j])):
                    if (data[j][x]>c): tmppass.append(feats[j][x])
                    else:              tmpfail.append(feats[j][x])
                bufpass.append(tmppass)
                buffail.append(tmpfail)
            feats_pass.append(bufpass)
            feats_fail.append(buffail)

        for fi in range(len(self._features_to_plot)):
            for ci in range(len(self._nnout_cuts)):
                plt.clf()
                bins=15
                for ip in range(2): # for pass and fail
                    for j in range(len(data)):
                        temp = np.array([])
                        pfstr = ""
                        stylestr = ""
                        doPlot = False
                        if (ip==0 and len(feats_pass[j][ci][:])>1): 
                            temp = np.array(feats_pass[j][ci])
                            temp = np.reshape(temp,(-1,len(self._features_to_plot)))
                            pfstr = " pass"
                            stylestr = "solid"
                            if (len(temp[:,fi])>50): doPlot = True
                        elif (ip==1 and len(feats_fail[j][ci][:])>1):
                            temp = np.array(feats_fail[j][ci])
                            temp = np.reshape(temp,(-1,len(self._features_to_plot)))
                            pfstr = " fail"
                            stylestr = "dashed"
                            if (len(temp[:,fi])>50): doPlot = True
                        if (doPlot and j<4): plt.hist(temp[:,fi],bins,log=False,histtype='step',
                                                      density=True,linestyle=stylestr, label=labels[j]+pfstr,
                                                      fill=False,range=self._features_range[fi],
                                                      color=self._colorlist[j])
                plt.legend(loc='best')
                plt.xlabel(self._features_to_plot[fi])
                plt.ylabel('Number of events (normalized)')
                plt.title('NeuralNet < '+str(self._nnout_cuts[ci])+' applied to test samples')
                plt.savefig(self._odir+"/"+self._key+"_"+self._features_to_plot[fi]+"_"+str(ci)+".pdf")
                plt.yscale('log')
                plt.savefig(self._odir+"/"+self._key+"_"+self._features_to_plot[fi]+"_"+str(ci)+"_log.pdf")
                plt.yscale('linear')
                plt.show()

    def plotROC(self,truth,scores,labels):
        plt.clf()
        for j in range(len(truth)):
            x,y,_ =roc_curve(truth[j],scores[j])
            auc = roc_auc_score(truth[j],scores[j])
            plt.plot(x,y,label='{}, AUC = {:.2f}'.format(labels[j],auc))
        plt.legend(loc='lower right')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.savefig("%s/%s_roc.pdf"%(self._odir,self._key))
        #plt.show()
    
    def test(self,model,X_test,Y_test,Y_adv_test,feat_test):
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
        for subsample in subsamples:
            ids=np.logical_and(Y_test==subsample[0],Y_adv_test==subsample[1])
            tmpdata = X_test[ids]
            tmpfeat = feat_test[ids]
            tmppred = model.predict(tmpdata)
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
        self.plotNNResponse(response_preds,labels)
        self.plotFeatResponse(response_preds,feat_preds,labels)
        self.plotROC(roc_true, roc_preds, roclabels)
    
    def plot_losses(self,history,losses,legends):
        plt.clf()
        for loss in losses:
            plt.plot(history.history[loss])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(legends, loc='upper left')
        plt.savefig("%s/%s_loss.pdf"%(self._odir,self._key))
        plt.show()

    def plotVariable(self,data,labels,variable,ranges):
        plt.clf()
        for i in range(0,len(data)):
            plt.hist(data[i], bins=30, fill=False,density=True,histtype='step',label=labels[i],range=ranges, linewidth=1.4)
        plt.legend(loc='best',prop={'size': 20})
        plt.xlabel(variable, fontsize=18)
        plt.ylabel('Number of events (normalized)', fontsize=18)
        #plt.title(variable, fontsize=18)
        plt.savefig("%s/%s_%s.pdf"%(self._odir,self._key,variable))
        plt.show()

    def regression(self,model,X,Y,Y_adv,Y_mh,Y_jetmet,feat,ipred=-1,targetname='met_pt',targetrange=(0,200)):
        labels     = ["h other","hWW","htau tau ","h glu glu","h cc"] #"hbb", "h ZZ"
        subsamples = [0.,2.,3.,4.,6.]
        response_preds = []
        target_preds = []
        mh_preds = []
        mhrec_preds = []
        for subsample in subsamples:
            if subsample==1: continue
            if ipred==-1:
                tmppred = np.array(model.predict(X[Y_adv==subsample]))
                tmpy = Y[Y_adv==subsample].flatten()
            else:
                tmppred = np.array(model.predict(X[Y_adv==subsample]))[ipred]
                tmpy = Y[ipred][Y_adv==subsample]
            response_preds.append(tmppred.flatten())
            target_preds.append(tmpy)
            '''
            if targetname!='genmh':
                mh_preds.append(Y_mh[Y_adv==subsample])
                if targetname='met_pt':
                    metpt = tmppred
                else:
                    metpt = Y_jetmet[4][Y_adv==subsample]
                tmpmhrec = reconstruct_mH_vals(Y_jetmet[0][Y_adv==subsample],
                                               Y_jetmet[1][Y_adv==subsample],
                                               Y_jetmet[2][Y_adv==subsample],
                                               Y_jetmet[3][Y_adv==subsample],
                                               metpt,
                                               Y_jetmet[5][Y_adv==subsample],
                                               Y_jetmet[6][Y_adv==subsample],
                                               Y_jetmet[7][Y_adv==subsample])
                mhrec_preds.append(tmpmhrec)
            '''
        print('now plotting prediction')
        self.plotVariable(response_preds,labels,'pred_%s'%targetname,targetrange)
        print('now plotting input')
        self.plotVariable(target_preds,labels,'input_%s'%targetname,targetrange)
        #print('now plotting features')
        #self.plotFeatResponse(response_preds,feat_preds,labels)
        
def plotSigvsBkg(y, var, X=None, xlabel='', ylabel='', label='mass',  odir='./', legend=True, bins=50, ax=None, save=True):
    if isinstance(var, int):
        assert X is not None, "Requested plot of integer variable with no feature array."
        var = X[:,var]
        pass
    sig = (y == 1)
    common = dict(bins=bins, alpha=0.5)
    
    if ax is None:
        _, ax = plt.subplots(figsize=(6,5))
        pass
    nsig = np.sum( sig)
    nbkg = np.sum(~sig)
    wsig = np.ones((nsig,)) / float(nsig)
    wbkg = np.ones((nbkg,)) / float(nbkg)
        
    ax.hist(var[ sig], weights=wsig, color='orange', label='Signal',     **common)
    ax.hist(var[~sig], weights=wbkg, color='blue', label='Background', **common)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or 'Fraction of jets')
    if legend:
        ax.legend()
        pass
    
    if save:
        plt.savefig(odir+"/"+label+"_sigvsbkg.pdf")
        plt.show()
    return ax

def plotInputs(y,x,names,ranges,label='inputs', odir='./'):
    fig, axes = plt.subplots(3, 4, figsize=(10,10), sharey=True)
    for ix, (ax, rg, nm) in enumerate(zip(axes.flatten(), ranges, names)):
        bins = np.linspace(*rg, num=50 + 1, endpoint=True)
        plotSigvsBkg(y, ix, x, bins=bins, xlabel='Feature {}: {:s}'.format(ix, nm), ylabel=' ', ax=ax, legend=(ix == 0), save=False)
        pass
    fig.tight_layout()
    axes[0][0].set_ylabel('Fraction jets')
    axes[1][0].set_ylabel('Fraction jets')
    fig.delaxes(axes.flatten()[-1])
    fig.show()
    fig.savefig(odir+"/"+label+"_sigvsbkg.pdf")

def plotRoc(y_true, y_preds, labels=None, legend=True, ax=None, save=False, label='inputs', odir='./'):
    if not isinstance(y_preds, list):
        y_preds = [y_preds]
        pass
    
    N = len(y_preds)
    assert N > 0, "[roc] No predictions provided"
    
    if labels is None:
        labels = [None for _ in range(N)]
    else:
        if isinstance(labels, str):
            labels = [labels]
            pass
        assert len(labels) == N, "[roc] Number of predictions ({}) and associated labels ({}) do not match.".format(N, len(labels))
        pass
    
    fprs, tprs = list(), list()
    for ix in range(N):
        
        if y_preds[ix] is None:
            fprs.append(None)
            tprs.append(None)
            continue
        
        fpr, tpr, _ = roc_curve(y_true, y_preds[ix])

        if auc(fpr, tpr) < 0.5:
            fpr = 1. - fpr
            tpr = 1. - tpr
            pass
        
        msk = (tpr > 0) & (fpr > 0)
        fpr = fpr[msk]
        tpr = tpr[msk]

        fprs.append(fpr)
        tprs.append(tpr)
        pass

    if ax is None:
        _, ax = plt.subplots(figsize=(6,5))
        pass
    
    ax.plot(tprs[0], 1./tprs[0], 'k:', label='Random guessing')
    for label, tpr, fpr in zip(labels, tprs, fprs):
        if tpr is None:
            ax.plot([0], [0])
        else:
            ax.plot(tpr, 1./fpr, label=label)
            pass
        pass
    ax.set_xlabel('Signal efficiency')
    ax.set_ylabel('Background rejection')
    ax.set_yscale("log", nonposy='clip')
    if legend:
        ax.legend()
        pass

    if save:
        plt.savefig(odir+"/"+label+"_rocall.pdf")
        plt.show()
    
    return None

def plotProfile (m, ys, labels=None, bins=40, ax=None, save=False, label='inputs', odir='./'):
    if isinstance(bins, int):
        bins = np.linspace(m.min(), m.max(), bins + 1, endpoint=True)
        pass
    
    if not isinstance(ys, list):
        ys = [ys]
        pass
    
    N = len(ys)
    centres = bins[:-1] + 0.5 * np.diff(bins)

    if labels is None:
        labels = [None for _ in range(N)]
    elif isinstance(labels, str):
        labels = [labels]
        pass
    
    assert len(labels) == N, "[profile] Number of observables ({}) and associated labels ({}) do not match.".format(N, len(labels))

    profiles = {ix: list() for ix in range(N)}
    means_NN  = list()
    means_ANN = list()
    for down, up in zip(bins[:-1], bins[1:]):
        msk = (m >= down) & (m < up)
        for ix, y in enumerate(ys):
            profiles[ix].append(y[msk].mean())
            pass
        pass
    
    if ax is None:
        _, ax = plt.subplots(figsize=(6,5))
        pass
    
    for ix in range(N):
        ax.plot(centres, profiles[ix], '.-', label=labels[ix])
        pass
    
    ax.set_xlabel('Jet mass [GeV]')
    ax.set_ylabel('Average value of observable')
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylim(0, 1)
    ax.legend()
    if save:
        plt.savefig(odir+"/"+label+"_profile.pdf")
        plt.show()
    
    return ax

def plotSculpting (m, y, preds, labels=None, effsig=0.5, bins=40, ax=None, save=False, label='inputs', odir='./'):
    # Check(s)
    if isinstance(bins, int):
        bins = np.linspace(m.min(), m.max(), bins + 1, endpoint=True)
        pass
    
    if not isinstance(preds, list):
        preds = [preds]
        pass
    
    N = len(preds)
    
    if labels is None:
        labels = [None for _ in range(N)]
    elif isinstance(labels, str):
        labels = [labels]
        pass
    
    assert len(labels) == N, "[sculpting] Number of observables ({}) and associated labels ({}) do not match.".format(N, len(labels))
    
    # ... labels...
    
    # Ensure axes exist
    if ax is None:
        _, ax = plt.subplots(figsize=(6,5))
        pass
    
    # Common definitions
    sig = (y == 1).ravel()
    common = dict(bins=bins, alpha=0.5)
    
    # Get weights
    nsig = np.sum( sig)
    nbkg = np.sum(~sig)
    wsig = np.ones((nsig,)) / float(nsig)
    wbkg = np.ones((nbkg,)) / float(nbkg)
    
    # Draw original mass spectrum, legend header
    ax.hist(m[ sig], color='red',  bins=bins, weights=wsig, label='Signal, incl.', histtype='step', lw=2)
    ax.hist(m[~sig], color='blue', bins=bins, weights=wbkg, label='Background, incl.', histtype='step', lw=2)
    ax.hist([0], weights=[0], color='black', label='Bkgds., $\\varepsilon_{{sig}} = {:.0f}\%$ cut'.format(effsig * 100.), **common)

    for pred, label in zip(preds, labels):
        
        # -- Get cut
        cut = np.percentile(pred[sig], effsig * 100.)
        msk = (~sig) & (pred > cut).ravel()  # Assuming signal towards larger values
        
        # -- Get weights
        nmsk = np.sum( msk)
        wmsk = np.ones((nmsk,)) / float(nmsk)
        
        # -- Plot
        ax.hist(m[msk],  weights=wmsk, label="   " + label, **common)
        pass
    
    ax.set_ylabel('Fraction of jets')
    ax.set_xlabel('Jet mass [GeV]')
    ax.set_yscale('log')
    ax.set_ylim(1.0E-03, 5.0E-01)
    ax.legend()

    if save:
        plt.savefig(odir+"/"+label+"_sculpting.pdf")
        plt.show()
    
    return ax
    
def plotOutputs(mass_test, Y_test, y_pred_NN, nnlabels=['NN classifier',None], label='outputs', odir='./'):
    # Create axes
    fig, axes = plt.subplots(2,2, figsize=(12,8))
    axes = axes.flatten()
    
    # (1) Stand-alone NN classifier observable distributions
    plotSigvsBkg(Y_test, y_pred_NN[0], xlabel=nnlabels[0], ax=axes[0],save=False)
    
    # (2) ROC curves
    plotRoc(Y_test, y_pred_NN, nnlabels, ax=axes[1], save=False)
    
    # (3) Profile of average classifier observable versus jet mass for background jets
    bkg = (Y_test == 0)
    plotProfile(mass_test[bkg], y_pred_NN[0][bkg], nnlabels[0], ax=axes[2], save=False)
    
    # (4) Jet mass distribution sculpting
    plotSculpting(mass_test, Y_test, y_pred_NN[0], nnlabels[0], ax=axes[3], save=False)
    
    fig.tight_layout()
    fig.savefig(odir+"/"+label+".pdf")

def plotPosterior(adv, m, z, nb_bins=100, title='', label='outputs', odir='./', ipred=-1):
    # Definitions
    scale = m.max()
    colours = ['r', 'g', 'b']
    zs = [0.2, 0.4, 0.8]
    tol = 0.05
    
    
    # Binning, scaled and not
    mt_pdf = np.linspace(0, 1., nb_bins + 1, endpoint=True)
    bins = mt_pdf * scale
    
    fig, ax = plt.subplots(1, 2, figsize=(10,4), sharey=True)

    # -- Left pane
    # Draw inclusive background jet distribution
    ax[0].hist(m, bins, density=1., alpha=0.3, color='black', label='Background, incl.')
    for col, z_ in zip(colours, zs):
        # Draw adversary posterior p.d.f. for classifier output `z`
        posterior = adv.predict([z_*np.ones_like(mt_pdf), mt_pdf])
        #print('posterior ',posterior)
        if ipred>-1:
            posterior= posterior[ipred]
        ax[0].plot(mt_pdf * scale, posterior / scale, color=col, label='z = {:.1f}'.format(z_))
        pass
    
    # -- Right pane
    ax[1].hist([0], bins, alpha=0.3, weights=[0], color='black', label='Background, $z_{NN}$-bin.')
    for col, z_ in zip(colours, zs):
        # Draw background jet distribution for classifier output `z`
        msk = np.abs(z - z_).ravel() < tol
        ax[1].hist(m[msk], bins, color=col, density=1., alpha=0.3, label='  {:.2f} < $z_{{NN}}$ < {:.2f}'.format(z_ - tol, z_ + tol))
        
        # Draw adversary posterior p.d.f. for classifier output `z`
        posterior = adv.predict([z_*np.ones_like(mt_pdf), mt_pdf])
        #print('posterior 2 ',posterior)
        if ipred>-1:
            posterior= posterior[ipred]
        ax[1].plot(mt_pdf * scale, posterior / scale, color=col, label='z = {:.1f}'.format(z_))
        pass

    ax[0].legend()
    ax[1].legend()
    ax[0].set_ylabel('Probability density')
    ax[0].set_xlabel('Jet mass [GeV]')
    ax[1].set_xlabel('Jet mass [GeV]')
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(odir+"/"+label+".pdf")
    return ax
