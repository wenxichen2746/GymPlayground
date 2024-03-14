#!/bin/bash

# Base configuration
env="CartPole-v1"
log_folder="trained_model_CP" 
#conf="configs/a2c.yml"
save_freq="15000"
n_timesteps="150000"
algo="ppo"
ent_coef='0.'

# Loop through seed values
for seed in {12..14}; do
    # Calculate experiment ID
    exp_id=$((10 + seed))

    # Construct command
    command="python3 -m rl_zoo3.train -P --env $env --log-folder $log_folder --n-eval-envs 1000 --save-freq $save_freq --n-timesteps $n_timesteps --algo $algo -params ent_coef:$ent_coef --seed $seed --study-name $exp_id"
    
    # Print the command (for debugging purposes)
    echo "Executing: $command"
    echo "-------Start Training-----"
    
    # Execute the command
    $command
done


