# Generalization performance of narrow one-hidden layer networks in the teacher-student setting

This project studies the generalization capabilities of narrow neural networks with a single hidden layer in a teacher-student setup.

We explore how architectural constraints and training dynamics affect learning and generalization, particularly in high-dimensional settings.

## Contents

- Experimental simulations using PyTorch
- Mathematica notebook used to solve the saddle point equation of the theoretical analysis
- Visualization of Free Energy, generalization error and overlap for Quadratic Activation function
- Comparisons between Numerics and Theory for activation functions (ReLU, erf, etc.)

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- PyTorch

## How to Run

1. Clone the repository
2. Install dependencies

3. To launch training runs with custom settings, use the provided .sh script. It automatically executes the Python training script with your chosen parameters and logs output to ./all_logfile/.

    Key Options:
        -	initialize_previousepochs: Resume training from previous results (True) or start fresh (False).
        -	bias_trick: Center the labels for improved training.
        -	wise_initialization: Start from a student close to the teacher (planted initialization).
        -	alpha_values, lr_list, T : Sets of hyperparameters over which to iterate.

## How to Use:

    Run the script with:

    bash run_experiment.sh

    This will run all combinations of parameters in the background and save logs for later review.



## Link to the Paper
