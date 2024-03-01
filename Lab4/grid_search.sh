#!/bin/bash

# Define the start and end values for the detector threshold
det_start=0.95
det_end=0.99

# Define the start and end values for the classifier threshold
clf_start=0.48
clf_end=0.51

# Define the increment value
increment_det=0.01
increment_clf=0.01

# Loop over the detector threshold values
for det in $(seq $det_start $increment_det $det_end)
do
    # Loop over the classifier threshold values
    for clf in $(seq $clf_start $increment_clf $clf_end)
    do
        # Print the current values
        echo "Testing with detector threshold: $det and classifier threshold: $clf"

        # Run the python script with the current threshold values
        python src/CHALL_AGC_FRbasicScript.py --detector_threshold $det --classifier_threshold $clf
    done
done
