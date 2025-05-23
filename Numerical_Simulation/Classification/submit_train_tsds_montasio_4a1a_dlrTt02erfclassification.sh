#! /bin/bash

# #SBATCH --job-name=com
# #SBATCH --time=00:20:00
# #SBATCH --mem=10G
# #SBATCH --cpus-per-task=4
# #SBATCH --partition=testing
# #SBATCH -o job-%j.out
# #SBATCH -e job-%j.err

# module load anaconda3
# Log file for output
# Create the directory if it doesn't exist
mkdir -p ./all_logfile

teacher_student="True"
N=500
nonlinearity="erf"
sigma2_y=0.333
num_test=1000
random_v=""
mean_field_v=""
learn_v=""
scale_gamma_T=""
use_bias_W=""
use_bias_v=""
num_epochs=300
print_every=10
#lr=0.001
gamma=0.0
K=100
T=(0.0001)
seed=8
seed_all=$seed
lr_list=(0.07)
momentum=0.
save_dir="results"
pertubation=0.001
perturb_amount=1e-5 
bias_trick="False"
initialize_previousepochs="False"
alpha_values=(1.0) # 2.5 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0)   #(4.0 5.0 6.0 7.0 8.0 9.0 10.0) (0.01 0.1 0.5 0.8 1.0 1.5 2.0 3.0 4.0 5.0) (20.0 18.0 16.0 14.0 12.0 10.0 8.0 6.0) (5.0 4.0 3.0 2.5 2.0 1.5 1.0 0.8 0.5 0.1 0.01)
#alpha_values2=(2.5) #(8.0 6.0 5.0 4.0 3.0 2.5 2.0) #(5.0 4.0 3.0 2.5 2.0 1.5 1.0 0.8 0.5 0.1 0.01)
wise_initialization="True"
LOG_FILE="./all_logfile/classification_script_output_teacher_student$teacher_student nonlinearity$nonlinearity bias_trick$bias_trick wise_initialization$wise_initialization N$N K$K K_teacher$K_teacher T$T gamma$gamma.log"


# Ensure the script runs in the background using nohup
{
    echo "Script started at $(date)"

    for T in "${T[@]}"; do
        for K in $K; do
            K_teacher=$K
            for gamma in $gamma; do
                for seed in $seed; do
                    for lr in "${lr_list[@]}"; do
                        for alpha in "${alpha_values[@]}"; do
                            # Execute the training script with the selected alpha in the background
                            python3 train_script_bayeserrorclassification.py \
                                --wise_initialization $wise_initialization\
                                --initialize_previousepochs $initialize_previousepochs\
                                --teacher_student $teacher_student\
                                --N $N \
                                --K $K \
                                --pertubation $pertubation \
                                --seed_all $seed_all\
                                --K_teacher $K_teacher \
                                --nonlinearity $nonlinearity \
                                --alpha $alpha \
                                --gamma $gamma \
                                --sigma2_y $sigma2_y \
                                --T $T \
                                --num_test $num_test \
                                $random_v $mean_field_v $learn_v $scale_gamma_T \
                                $use_bias_W $use_bias_v \
                                --num_epochs $num_epochs \
                                --print_every $print_every \
                                --lr $lr \
                                --momentum $momentum \
                                --seed $seed \
                                --bias_trick $bias_trick\
                                --save_dir $save_dir \
                                --perturb_amount $perturb_amount &
                                pid=$!  # Capture the PID of the background process
                                echo "Started process with PID $pid for teacher_student=$teacher_student nonlinearity=$nonlinearity wise_initialization=$wise_initialization T=$T K=$K N=$N gamma=$gamma K_teacher=$K_teacher Gamma=$gamma alpha=$alpha lr=$lr"
                        done
                    done
                done
            done
        done
    done

} >> "$LOG_FILE" 2>&1 & 

echo "Script is running in the background. Check the log file: $LOG_FILE"


wait  # Wait for all background processes to finish
