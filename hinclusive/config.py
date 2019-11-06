def append_to(element, to=None):
    if to is None:
        to = []
    to.append(element)
    return to

def extend_to(element, to=None):
    if to is None:
        to = []
    to.extend(element)
    return to

lFeatures = ['h_decay_id1','h_decay_id11','h_decay_id21','j_pt','j_eta','j_phi','j_mass','j_mass_mmdt','j_tau21_b1',
             'j_mass_trim','j_mass_rsdb1','j_mass_sdb1','j_mass_prun','j_mass_sdb2','j_mass_sdm1',
]
lProc     = ['procid']
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
hids = [0.,1.,2.,3.,4.,5.,6.]

idconv = {11.:1, 12.:2, 13.:3, 22.:4, 130.:5, 211.:6, 
          310.:7, 321.:8, 2112.:9, 2212.:10, 
          3112.:11, 3122.:12, 3222.:13, 3312.:14, 3322.:15, 3334.:16, 
          -11.:17, -12.:18, -13.:19, -22.:20, -130.:21, -211.:22, 
          -310.:23, -321.:24, -2112.:25, -2212.:26, 
          -3112.:27, -3122.:28, -3222.:29, -3312.:30, -3322.:31, -3334.:32, 0.:0}
nIDs = 33

lECFs_noNaNs = lECFs[:len(lECFs)-4]

pPhilOldDir = '/eos/cms/store/cmst3/group/exovv/precision/'
pPhilDir = '/eos/cms/store/cmst3/group/monojet/precision/'
pMineDir = '/eos/cms/store/group/phys_exotica/dijet/dazsle/precision/'
pTrackDir = '/eos/cms/store/group/phys_exotica/dijet/dazsle/TrackObservablesStudy/'
lDict = {
    'qcd1000to1500': {'files': [['QCD_HT1000to1500_tarball.tar.xz',pPhilDir,'ntupler'],
                                ['QCD_HT1000to1500_tarball.tar.xz',pPhilOldDir,'ntupler'],
                                ['QCD_HT1000to1500_tarball.tar.xz',pMineDir,'metntupler']],
                      'sig': 0,
                      'xs': 1207},
    'qcd1500to2000': {'files': [['QCD_HT1500to2000_tarball.tar.xz',pPhilDir,'ntupler'],
                                ['QCD_HT1500to2000_tarball.tar.xz',pPhilOldDir,'ntupler'],
                                ['QCD_HT1500to2000_tarball.tar.xz',pMineDir,'metntupler']],
                      'sig': 0,
                      'xs': 119.9},
    'qcd2000toInf': {'files': [['QCD_HT2000toInf_tarball.tar.xz',pPhilDir,'ntupler'],
                               ['QCD_HT2000toInf_tarball.tar.xz',pPhilOldDir,'ntupler'],
                               ['QCD_HT2000toInf_tarball.tar.xz',pMineDir,'metntupler']],
                     'sig': 0,
                     'xs':25.24},
    'qcd700to1000': {'files': [['QCD_HT700to1000_tarball.tar.xz',pPhilDir,'ntupler'],
                               ['QCD_HT700to1000_tarball.tar.xz',pPhilOldDir,'ntupler'],
                               ['QCD_HT700to1000_tarball.tar.xz',pMineDir,'metntupler']],
                     'sig': 0,
                     'xs':6831},
    'qcd500to700': {'files': [['QCD_HT500to700_tarball.tar.xz',pPhilDir,'ntupler',
                               'QCD_HT500to700_tarball.tar.xz',pPhilOldDir,'ntupler']],
                     'sig': 0,
                     'xs':32100},
    'qcd300to500': {'files': [['QCD_HT300to500_tarball.tar.xz',pPhilDir,'ntupler',
                               'QCD_HT300to500_tarball.tar.xz',pPhilOldDir,'ntupler']],
                    'sig': 0,
                    'xs':347700},
    'wqq': {'files': [['wqq_pt300_100_tarball.tar.xz',pPhilDir,'ntupler']],
            'sig': 0,
            'xs':36.19},
    'zqq': {'files': [['zqq_pt300_100_tarball.tar.xz',pPhilDir,'ntupler']],
            'sig': 0,
            'xs':15.28},
    'tqq': {'files': [['TT_hdamp_NNPDF31_NNLO_had.tgz',pMineDir,'metntupler']],
            'sig': 0,
            'xs':730.0*1.139397},
    'ggh': {'files': [['ggh012j_ptj20_nocard_tarball.tar.xz_h12j',pPhilDir,'metntupler'],
                      ['ggh012j_ptj20_nocard_tarball.tar.xz_h12j',pMineDir,'metntupler']],
            'sig': 1,
            'xs':0.1677*5},
    'tth': {'files': [['ttH_inclusive_hdamp_NNPDF31_13TeV_M125.tgz',pPhilDir,'ntuplermethid']],
            'sig': 2,
            'xs':0.5071},
    'vbf': {'files': [['VBF_H_NNPDF31_13TeV_M125_slc6_amd64_gcc630_CMSSW_9_3_0.tgz',pPhilDir,'ntuplermethid']],
            'sig': 2,
            'xs':3.748},
    'whp': {'files': [['HWJ_slc6_amd64_gcc630_CMSSW_9_3_0_HWminusJ_HanythingJ_NNPDF31_13TeV_M125_Vinclusive.tgz',pPhilDir,'ntuplermethid']],
            'sig': 2,
            'xs':0.8400},
    'whm': {'files': [['HWJ_slc6_amd64_gcc630_CMSSW_9_3_0_HWplusJ_HanythingJ_NNPDF31_13TeV_M125_Vinclusive.tgz',pPhilDir,'ntuplermethid']],
            'sig': 2,
            'xs':0.5328},
    'zh': {'files': [['HZJ_HanythingJ_NNPDF31_13TeV_M125.tgz',pPhilDir,'ntuplermethid']],
           'sig': 2,
           'xs':0.8839},
    'zpqq': {'files': [['Vector_Dijet_LO_Mphi125_slc6_amd64_gcc481_CMSSW_7_1_30_tarball.tar.xz',pPhilDir,'ntuplermethid']],
             'sig': 0,
             'xs':14.515932},
    }
