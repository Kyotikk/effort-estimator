#!/bin/bash
# Quick script to run pipeline for all datasets

set -e

echo "======================================================================"
echo "RUNNING PIPELINES FOR ALL DATASETS"
echo "======================================================================"

success_count=0
fail_count=0

data_root="/Users/pascalschlegel/data/interim"

# Find latest file (lexicographic) matching a pattern under a folder
find_latest() {
    local pattern="$1"
    # shellcheck disable=SC2086
    ls $pattern 2>/dev/null | sort | tail -n 1
}

# Auto-detect date from available corsano_bioz_acc files for patient/condition
detect_date() {
    local patient="$1"
    local condition="$2"
    local acc_latest
    acc_latest=$(find_latest "$data_root/$patient/$condition/corsano_bioz_acc/*.csv*")
    if [[ -n "$acc_latest" ]]; then
        basename "$acc_latest" | sed 's/\.csv.*$//'
    else
        echo "2025-12-04"  # fallback to previous default
    fi
}

# Auto-detect ADL filename (csv or csv.gz) if present
detect_adl() {
    local patient="$1"
    local condition="$2"
    local adl
    adl=$(find_latest "$data_root/$patient/$condition/scai_app/ADLs_1.csv*")
    if [[ -n "$adl" ]]; then
        basename "$adl"
    else
        echo "ADLs_1.csv"
    fi
}

# Calculate ADL offset (hours) based on min times in ADL vs HR data
detect_adl_offset() {
    local patient="$1"
    local condition="$2"
    .venv/bin/python << PYEOF
import gzip, pandas as pd, sys
from pathlib import Path

adl_path = Path("$data_root/$patient/$condition/scai_app/ADLs_1.csv.gz")
hr_path = Path("$data_root/$patient/$condition/vivalnk_vv330_heart_rate/data_1.csv.gz")

if adl_path.exists() and hr_path.exists():
    try:
        adl_df = pd.read_csv(gzip.open(adl_path, 'rt'))
        hr_df = pd.read_csv(gzip.open(hr_path, 'rt'))
        # offset = (HR_min - ADL_min) so that when added to ADL times, they align
        offset_sec = (hr_df['time'].min() - adl_df['time'].min())
        offset_hours = offset_sec / 3600.0
        print(f"{offset_hours:.1f}")
    except Exception as e:
        print("0", file=sys.stderr)
else:
    print("0", file=sys.stderr)
PYEOF
}

# Process each dataset
for patient in parsingsim3 parsingsim4 parsingsim5; do
    for condition in sim_healthy sim_elderly sim_severe; do
        full_condition="${condition}${patient: -1}"  # e.g., sim_healthy3
        date=$(detect_date "$patient" "$full_condition")
        adl_file=$(detect_adl "$patient" "$full_condition")
        adl_offset=$(detect_adl_offset "$patient" "$full_condition")
        
        echo ""
        echo "======================================================================"
        echo "Processing: $patient / $full_condition (date: $date, ADL offset: ${adl_offset}h)"
        echo "======================================================================"
        
        # Create temporary config
        config_file="config/pipeline_${full_condition}.yaml"
        
        sed -e "s|parsingsim3/sim_elderly3|$patient/$full_condition|g" \
            -e "s|parsingsim3_sim_elderly3|${patient}_${full_condition}|g" \
            -e "s|2025-12-04|$date|g" \
            -e "s|ADLs_1.csv|$adl_file|g" \
            -e "s|adl_offset_hours: 0|adl_offset_hours: $adl_offset|g" \
            config/pipeline.yaml > "$config_file"
        
        echo "✓ Created $config_file"
        
        # Run pipeline
        if .venv/bin/python run_pipeline.py "$config_file"; then
            echo "✓ SUCCESS: $patient/$full_condition"
            ((success_count++))
        else
            echo "✗ FAILED: $patient/$full_condition"
            ((fail_count++))
        fi
    done
done

total=$((success_count + fail_count))

echo ""
echo "======================================================================"
echo "SUMMARY"
echo "======================================================================"
echo "Total: $total datasets"
echo "  ✓ Successful: $success_count"
echo "  ✗ Failed: $fail_count"

if [ $success_count -gt 0 ]; then
    echo ""
    echo "======================================================================"
    echo "NEXT STEP:"
    echo "======================================================================"
    echo "Run: .venv/bin/python train_hrv_recovery_multidataset.py"
    echo "This will combine all $success_count datasets and train a model"
fi
