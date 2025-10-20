#!/bin/bash

# Define the specific order of model numbers to run
model_numbers=(32)

# Loop through the models
for model_num in "${model_numbers[@]}"
do
    # Define the Python script command and output file
    script_command="nohup python3 inference.py ${model_num}"
    output_file="outputTXTs/outputInferenceModel${model_num}.txt"

    # Run the Python script with nohup and redirect output
    $script_command > "$output_file" 2>&1 &

    echo "Started inference for model $model_num, output redirected to $output_file"
done
