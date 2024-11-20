#!/bin/bash

# Move runner files to src directory
mv pipeline_runner.py src/
mv advanced_pipeline_runner.py src/
mv run_all_ablation_studies.py src/
mv run_preprocessing_ablation.py src/
mv run_ensemble_evaluation.py src/
mv run_final_analysis.py src/
mv run_analysis.py src/
mv generate_report.py src/

# Update the run_pipeline.sh script
cat > run_pipeline.sh << 'EOL'
#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up logging - will save to logs/pipeline_run_TIMESTAMP.log
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/pipeline_run_${TIMESTAMP}.log"

# Function to log messages to both console and file
log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

# Start logging
log "Pipeline execution started at $(date)"
log "Logging to: $LOG_FILE"

# Setup project structure
log "\nSetting up project structure..."
python setup_project.py 2>&1 | tee -a "$LOG_FILE"

# Run the complete pipeline in order
log "\nStarting pipeline execution..."

# Run pipelines from src directory
PYTHONPATH=. python -m src.pipeline_runner 2>&1 | tee -a "$LOG_FILE" && \
PYTHONPATH=. python -m src.advanced_pipeline_runner 2>&1 | tee -a "$LOG_FILE" && \
PYTHONPATH=. python -m src.run_all_ablation_studies 2>&1 | tee -a "$LOG_FILE" && \
PYTHONPATH=. python -m src.run_preprocessing_ablation 2>&1 | tee -a "$LOG_FILE" && \
PYTHONPATH=. python -m src.run_ensemble_evaluation 2>&1 | tee -a "$LOG_FILE" && \
PYTHONPATH=. python -m src.run_final_analysis 2>&1 | tee -a "$LOG_FILE" && \
PYTHONPATH=. python -m src.run_analysis 2>&1 | tee -a "$LOG_FILE"

# Check results
log "\nChecking generated files..."
log "\nReports directory:"
ls -R reports/ 2>&1 | tee -a "$LOG_FILE"
log "\nResults directory:"
ls -R results/ 2>&1 | tee -a "$LOG_FILE"
log "\nVisualizations directory:"
ls -R visualizations/ 2>&1 | tee -a "$LOG_FILE"

log "\nPipeline execution completed at $(date)!"

# Print final message
log "\nExecution log saved to: $LOG_FILE"
EOL

# Make the script executable
chmod +x run_pipeline.sh

echo "Files reorganized and run_pipeline.sh updated!"