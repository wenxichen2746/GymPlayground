#!/bin/bash

# Base configuration
env="CarRacing-v2"
env_kwargs="continuous:False"
log_folder="GymPlayground/trained_model"  # Notice the change from '//' to '/'
conf="GymPlayground/configs/a2c.yml"
save_freq="20"
n_timesteps="40"
algo="a2c"

# Loop through ent_coef values
for ent_coef in 0. 0.0001 0.01; do
    # Loop through seed values
    for seed in {0..9}; do
        # Calculate experiment ID
        exp_id=$((100 + ent_coef * 10 + seed))

        # Construct command
        command="python rl_zoo3.train.py --env $env --env-kwargs $env_kwargs --log-folder $log_folder --conf $conf --save-freq $save_freq --n-timesteps $n_timesteps --algo $algo -params ent_coef:$ent_coef --seed $seed --study-name $exp_id"
        
        # Print the command (for debugging purposes)
        echo "Executing: $command"
        echo "-------Start Training-----"
        
        # Execute the command
        $command
    done
done
