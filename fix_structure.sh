#!/bin/bash

# Move files to correct locations
mv pipeline_runner.py src/
mv advanced_pipeline_runner.py src/
mv run_all_ablation_studies.py src/
mv run_preprocessing_ablation.py src/
mv run_ensemble_evaluation.py src/
mv run_final_analysis.py src/
mv run_analysis.py src/
mv generate_report.py src/
mv analysis_report.py src/

# Fix import statements
find src/ -type f -name "*.py" -exec sed -i '' 's/from pipeline_runner/from src.pipeline_runner/g' {} \;
find src/ -type f -name "*.py" -exec sed -i '' 's/from config/from src.config/g' {} \;

# Make config.py a module
touch src/__init__.py

echo "✓ Directory structure fixed!"
