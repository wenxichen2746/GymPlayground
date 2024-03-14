#!/bin/bash

# Base configuration
env="CarRacing-v2"
env_kwargs="continuous:False"
log_folder="trained_model_CR"  # Notice the change from '//' to '/'
conf="configs/a2c.yml"
save_freq="20000"
n_timesteps="2000000"
algo="a2c"
enti=0
# Loop through ent_coef values
for ent_coef in 0.; do
    # Loop through seed values
    for seed in {0..9}; do
        # Calculate experiment ID
        exp_id=$((100 + enti* 10 + seed))

        # Construct command
        command="python3 -m rl_zoo3.train -P --env $env --env-kwargs $env_kwargs --n-eval-envs 10000 --log-folder $log_folder --conf $conf --save-freq $save_freq --n-timesteps $n_timesteps --algo $algo -params ent_coef:$ent_coef --seed $seed --study-name $exp_id"
        
        # Print the command (for debugging purposes)
        echo "Executing: $command"
        echo "-------Start Training-----"
        
        # Execute the command
        $command
    done
    ((enti++))
done
