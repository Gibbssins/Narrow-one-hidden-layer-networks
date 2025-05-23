#!/bin/bash

# Python script path
PYTHON_SCRIPT="ps_plot_save_data.py"

# Duration to run the script (in seconds)
TOTAL_RUNTIME=$((10)) 
#TOTAL_RUNTIME=$((10 * 3600))  # one day
#TOTAL_RUNTIME=$((1 * 14 * 3600))  # Two days
START_TIME=$(date +%s)

# Learning rate values
lr_values=(0.07)

# Log file for output
LOG_FILE="./script_output.log"

# Ensure the script runs in the background using nohup
{
    echo "Script started at $(date)"

    # Main loop
    while true; do
        CURRENT_TIME=$(date +%s)
        ELAPSED_TIME=$((CURRENT_TIME - START_TIME))

        # Break the loop if the total runtime exceeds 2 days
        if (( ELAPSED_TIME >= TOTAL_RUNTIME )); then
            echo "Two days have passed. Exiting..."
            break
        fi

        # Run the Python script for each learning rate
        for lr in "${lr_values[@]}"; do
            echo "$(date +"%Y-%m-%d %H:%M:%S") - Running script with lr=$lr"
            python3 "$PYTHON_SCRIPT" --lr "$lr" &
            pid=$!  # Capture the PID of the background process
            echo "Started process with PID $pid for lr=$lr"
        done

        # Wait for all background processes to finish
        wait

        # Check the exit status of the background processes
        if [ $? -ne 0 ]; then
            echo "Error: One or more Python scripts failed. Exiting..."
            exit 1
        fi

        echo "$(date +"%Y-%m-%d %H:%M:%S") - Python script executed successfully for all learning rates."

        # Wait for one hour before the next execution 3600
        sleep 1800
    done

    echo "Script ended at $(date)"
} >> "$LOG_FILE" 2>&1 &

echo "Script is running in the background. Check the log file: $LOG_FILE"