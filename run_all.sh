#!/bin/bash

# Check if monitor script exists
if [ ! -f "./monitor.sh" ]; then
  echo "Error: monitor.sh not found."
  exit 1
fi

# Check if inference executable exists
if [ ! -f "./build/inference" ]; then
  echo "Error: inference executable not found."
  exit 1
fi

# Paths to model files
model1="./model/yolov8n_352.xml"
model2="./model/yolov8n_pose_320_fp16.xml"

# Check if model files exist
if [ ! -f "$model1" ] || [ ! -f "$model2" ]; then
  echo "Error: Model files not found."
  exit 1
fi

# List of frequencies to set
frequencies=(1.50GHz 1.40GHz 1.30GHz 1.20GHz 1.10GHz 1000MHz 900MHz 800MHz 700MHz 600MHz)
# Outer loop for CPU frequencies
for freq in "${frequencies[@]}"; do
  echo -e "\nSetting CPU frequency to $freq"
  sudo cpufreq-set -u $freq -d $freq
  
  # Check if frequency was set correctly
  current_freq=$(cpufreq-info -f)
  echo "Current frequency: $current_freq"

  # Create output directory for the current frequency if it doesn't exist
  output_dir="./output/${freq}"
  if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
  fi
  


  # Inner loop for thread counts
  for nthreads in {4..1}; do
    echo "Running with $nthreads threads at $freq..."
    output_file="./output/${freq}/time_results_${nthreads}.json"
    ./monitor.sh "${output_dir}/monitor_${freq}_${nthreads}threads.csv" ./build/inference "$model1" "$model2" "$output_file" "$nthreads" 
    sleep 60
  done
  sleep 60
done

echo "All runs completed."
