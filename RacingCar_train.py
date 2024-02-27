from scripts.roo_train_utilis import *


args=get_args()

config={
    'env':'CarRacing-v2',
    'log_folder':f'trained_model/CarRacing/entcoef{0}',

    'no_render':True,
    'env-kwargs': 'continuous:False',
    'domain_randomize':False,
    'save-freq':1e4,
    'n_timesteps':2e4,

}

for key,value in config.items():
    setattr(args,key,value)

ent_coef=0.
setattr(args,'exp_id',101)
setattr(args,'hyperparams',{'ent_coef':ent_coef})

train(args)