#!/bin/bash

# Function to fix imports in a file
fix_imports() {
    local file=$1
    echo "Fixing imports in $file..."

    # Create temp file
    temp_file="${file}.tmp"
    touch "$temp_file"

    # Read file line by line and fix imports
    while IFS= read -r line || [ -n "$line" ]; do
        # Fix various import patterns
        if [[ $line == *"from pipeline_runner import"* ]]; then
            line="${line/from pipeline_runner/from src.pipeline_runner}"
        fi
        if [[ $line == *"from config import"* ]]; then
            line="${line/from config/from src.config}"
        fi
        if [[ $line == *"from analysis_report import"* ]]; then
            line="${line/from analysis_report/from src.analysis_report}"
        fi

        # Write the line to temp file
        echo "$line" >> "$temp_file"
    done < "$file"

    # Replace original with fixed version
    mv "$temp_file" "$file"
    echo "✓ Fixed imports in $file"
}

# Make sure analysis_report.py is in src directory
if [ -f "analysis_report.py" ]; then
    mv analysis_report.py src/
    echo "✓ Moved analysis_report.py to src/"
fi

# Create empty __init__.py files where needed
mkdir -p src/final_analysis src/models src/visualization
touch src/__init__.py src/final_analysis/__init__.py src/models/__init__.py src/visualization/__init__.py

# Fix imports in all Python files
find src -type f -name "*.py" | while read -r file; do
    fix_imports "$file"
done

echo "✓ All import paths have been fixed!"