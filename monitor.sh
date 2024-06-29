#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <output_file> <program_name> [program_args...]"
  exit 1
fi

# The output CSV file
OUTPUT_FILE="$1"

# The name of the program to monitor
PROGRAM_NAME="$2"

# The arguments for the program
PROGRAM_ARGS="${@:3}"

# Function to get the CPU usage of the program
get_cpu_usage() {
  top -b -n 2 -d 0.2 -p "$PROGRAM_PID" | tail -1 | awk -v pid="$PROGRAM_PID" '$1 == pid {print $9}'
}

# Function to get the CPU temperature
get_cpu_temp() {
  sensors | awk '/^temp1:/{print $2}' | tr -d '+deg C'
}

# Function to get the RAM usage of the program
get_ram_usage() {
  top -b -n 1 -p "$PROGRAM_PID" | grep "$PROGRAM_PID" | awk '{print $6}' | tr -d 'K'
}

# Initialize the CSV file with headers
echo "Timestamp,CPU Usage (%),CPU Temperature (deg C),RAM Usage (KB)" > $OUTPUT_FILE

# Launch the program in the background
$PROGRAM_NAME $PROGRAM_ARGS &

# Get the PID of the launched program
PROGRAM_PID=$!

# Function to clean up on script exit
cleanup() {
  echo "Terminating the monitored program..."
  kill $PROGRAM_PID
}

# Trap the script exit to run the cleanup function
trap cleanup EXIT

# Loop to monitor the program
while kill -0 $PROGRAM_PID > /dev/null 2>&1; do
  # Get current timestamp
  TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

  # Get CPU usage
  CPU_USAGE=$(get_cpu_usage)

  # Get CPU temperature
  CPU_TEMP=$(get_cpu_temp)

  # Get RAM usage
  RAM_USAGE=$(get_ram_usage)

  # Append the data to the CSV file
  echo "$TIMESTAMP,$CPU_USAGE,$CPU_TEMP,$RAM_USAGE" >> $OUTPUT_FILE

  # Wait for 1 second before checking again
  sleep 1
done
