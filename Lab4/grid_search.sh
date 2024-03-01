#!/bin/bash

# Define the start and end values for the detector threshold
det_start=0.5
det_end=0.95

# Define the start and end values for the classifier threshold
clf_start=0.1
clf_end=0.6

# Define the increment value
increment=0.05

# Loop over the detector threshold values
for det in $(seq $det_start $increment $det_end)
do
    # Loop over the classifier threshold values
    for clf in $(seq $clf_start $increment $clf_end)
    do
        # Print the current values
        echo "Testing with detector threshold: $det and classifier threshold: $clf"

        # Run the python script with the current threshold values
        python src/CHALL_AGC_FRbasicScript.py --detector_threshold $det --classifier_threshold $clf
    done
done