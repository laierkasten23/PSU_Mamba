#!/bin/bash

# Batch script to run dataset complexity analysis
# Usage: ./run_analysis.sh [simple|full]

echo "Dataset Complexity Analysis - Choroid Plexus Segmentation"
echo "=========================================================="

# Check if required packages are installed
python3 -c "import sklearn, nibabel, pandas, matplotlib, seaborn, scipy, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install -r requirements_complexity_analysis.txt
fi

# Default to simple analysis
ANALYSIS_TYPE=${1:-simple}

if [ "$ANALYSIS_TYPE" = "full" ]; then
    echo "Running comprehensive complexity analysis..."
    python3 dataset_complexity_analysis.py
elif [ "$ANALYSIS_TYPE" = "simple" ]; then
    echo "Running K-means clustering analysis..."
    python3 simple_kmeans_analysis.py
elif [ "$ANALYSIS_TYPE" = "test" ]; then
    echo "Running dataset validation test..."
    python3 test_dataset.py
else
    echo "Usage: $0 [simple|full|test]"
    echo "  simple: K-means clustering analysis (default)"
    echo "  full:   Comprehensive complexity analysis"
    echo "  test:   Validate dataset access"
    exit 1
fi

echo ""
echo "Analysis complete!"
echo "Check generated files for results."
