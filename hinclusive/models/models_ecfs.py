models = {'ecfs_adv-mass-v1': 'model_adversary_v1eval',
          'ecfs_adv-rho-v5': 'model_adversary_v5',
          'ecfs_adv-decay': 'model_classifier_advdecay',
          'ecfs_adv-mass-kl': 'model_classifiercombined',
          'ecfs_adv-mass-kl-decay': 'model_combineddecayadv_20ep',
          'ecfs_adv-mass-kl-decay-v1': 'model_combineddecayadv_30ep',
          'ecfs_classifier': 'model_classifier',
          'ecfs_adv-decay-v1': 'model_adversary_decay_lwr0001',
          'ecfs_adv-massdecay': 'model_adversary_mass_decay',
          }

modelsorder = ['ecfs_classifier','ecfs_adv-decay','ecfs_adv-decay-v1',
               'ecfs_adv-mass-kl','ecfs_adv-rho-v5',
               'ecfs_adv-mass-kl-decay-v1','ecfs_adv-massdecay',
]
