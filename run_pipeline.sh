#!/bin/bash

# Color definitions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up logging - will save to logs/pipeline_run_TIMESTAMP.log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/pipeline_run_${TIMESTAMP}.log"

# Function to log messages to both console and file
log() {
    echo -e "${2:-}$1${NC}" | tee -a "$LOG_FILE"
}

# Function to run a pipeline step and check for errors
run_step() {
    step_name="$1"
    command="$2"

    log "\n${step_name}..." "${YELLOW}"
    if eval "$command" 2>&1 | tee -a "$LOG_FILE"; then
        log "✓ ${step_name} completed successfully" "${GREEN}"
        return 0
    else
        log "✗ ${step_name} failed" "${RED}"
        return 1
    fi
}

# Start logging
log "Pipeline execution started at $(date)" "${GREEN}"
log "Logging to: ${LOG_FILE}" "${GREEN}"

# Setup project structure
run_step "Setting up project structure" "python setup_project.py" || exit 1

# Check for required data files
log "\nChecking for required data files..." "${YELLOW}"
required_files=(
    "data/solar_data/Solar_Energy_Production.csv"
    "data/solar_data/Solar_Power_Plant_Data.csv"
    "data/Renewable Energy World Wide 1965-2022"
)

missing_files=false
for file in "${required_files[@]}"; do
    if [ ! -e "$file" ]; then
        log "✗ Missing required file: $file" "${RED}"
        missing_files=true
    else
        log "✓ Found required file: $file" "${GREEN}"
    fi
done

if [ "$missing_files" = true ]; then
    log "\n✗ Error: Missing required data files. Please add them before continuing." "${RED}"
    exit 1
fi

# Run the complete pipeline in order
log "\nStarting pipeline execution..." "${GREEN}"

# Array of pipeline steps
declare -a pipeline_steps=(
    "Running basic pipeline#python -m src.pipeline_runner"
    "Running advanced pipeline#python -m src.advanced_pipeline_runner"
    "Running ablation studies#python -m src.run_all_ablation_studies"
    "Running preprocessing ablation#python -m src.run_preprocessing_ablation"
    "Running ensemble evaluation#python -m src.run_ensemble_evaluation"
    "Running final analysis#python -m src.run_final_analysis"
    "Generating final report#python -m src.run_analysis"
)

# Run each step
for step in "${pipeline_steps[@]}"; do
    IFS='#' read -r step_name step_command <<< "$step"
    if ! run_step "$step_name" "PYTHONPATH=. $step_command"; then
        log "\n✗ Pipeline failed at step: $step_name" "${RED}"
        log "Check the log file for details: $LOG_FILE" "${YELLOW}"
        exit 1
    fi
done

# Check results
log "\nChecking generated files..." "${YELLOW}"

# Function to check directory contents
check_directory() {
    local dir=$1
    local dir_name=$2

    log "\n${dir_name} directory:" "${YELLOW}"
    if [ -d "$dir" ]; then
        if [ "$(ls -A $dir)" ]; then
            ls -R "$dir" 2>&1 | tee -a "$LOG_FILE"
            log "✓ ${dir_name} files generated successfully" "${GREEN}"
        else
            log "⚠ ${dir_name} directory is empty" "${YELLOW}"
        fi
    else
        log "✗ ${dir_name} directory not found" "${RED}"
    fi
}

# Check each output directory
check_directory "reports" "Reports"
check_directory "results" "Results"
check_directory "visualizations" "Visualizations"

# Final status
log "\nPipeline execution completed at $(date)!" "${GREEN}"
log "✓ All steps completed successfully" "${GREEN}"
log "Execution log saved to: ${LOG_FILE}" "${YELLOW}"

# Print a summary of generated artifacts
log "\nGenerated Artifacts Summary:" "${GREEN}"
log "- Reports: $(find reports -type f | wc -l) files" "${NC}"
log "- Results: $(find results -type f | wc -l) files" "${NC}"
log "- Visualizations: $(find visualizations -type f | wc -l) files" "${NC}"