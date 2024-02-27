from scripts.roo_train_utilis import *


args=get_args()

config={
    'env':'CarRacing-v2',
    'log_folder':f'/trained_model',
    'no_render':True,
    'env_kwargs': {'continuous':False,'domain_randomize':False},
    'save_freq':2e4,
    'n_timesteps':4e6,

}

for key,value in config.items():
    #if hasattr(args,key):
        #print('Update',key,':',value)
    setattr(args,key,value)

for algo in ['a2c']:
    setattr(args,'algo',algo)
    
    for enti,ent_coef in enumerate([0., 0.0001,0.01]):

        #ent_coef=0.
        setattr(args,'hyperparams',{'ent_coef':ent_coef})
        print('args:',args)

        for seed in range(5):
            setattr(args,'seed',seed)
            setattr(args,'exp_id',enti*10+seed)
            train(args)