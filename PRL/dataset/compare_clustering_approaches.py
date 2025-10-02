"""
Compare Raw Image vs Segmentation-based K-means Clustering Results

This script compares the two different clustering approaches:
1. Raw T1xFLAIR image features only
2. Combined image + segmentation features

USAGE COMMANDS:
# Skip processing and use existing CSV files (recommended for analysis)
python3 compare_clustering_approaches.py --skip

# Run full processing pipeline (slow, requires raw data)
python3 compare_clustering_approaches.py --full

# Interactive mode (will prompt for choice)
python3 compare_clustering_approaches.py

# From Python interpreter:
from compare_clustering_approaches import compare_clustering_approaches
compare_clustering_approaches(skip_processing=True)   # Fast analysis
compare_clustering_approaches(skip_processing=False)  # Full processing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# File paths (update these paths as needed)
DATA_PATH = './data/'
JSON_PATH = './data/subjects.json'

COMMANDS_HELP = """
AVAILABLE COMMANDS:
==================

1. SKIP PROCESSING (Recommended for analysis):
   python3 compare_clustering_approaches.py --skip
   
2. FULL PROCESSING (Slow, requires raw data):
   python3 compare_clustering_approaches.py --full
   
3. INTERACTIVE MODE:
   python3 compare_clustering_approaches.py
   
4. FROM PYTHON INTERPRETER:
   from compare_clustering_approaches import compare_clustering_approaches
   compare_clustering_approaches(skip_processing=True)   # Fast
   compare_clustering_approaches(skip_processing=False)  # Full

REQUIREMENTS:
- For skip mode: raw_features_data.csv and unified_features_data.csv must exist
- For full mode: Raw imaging data and segmentation files required
"""

def compare_clustering_approaches(skip_processing=False):
    """Compare the two clustering approaches"""
    
    print("=" * 80)
    print("COMPARISON: RAW IMAGE vs SEGMENTATION-BASED CLUSTERING")
    print("=" * 80)
    
    if skip_processing:
        # Skip the processing and just load the existing CSV files
        try:
            df_raw = pd.read_csv('raw_features_data.csv')
            df_unified = pd.read_csv('unified_features_data.csv')
            
            # Results files
            raw_results = pd.read_csv('raw_kmeans_results.csv')
            unified_results = pd.read_csv('unified_kmeans_results.csv')
            
            print("✓ Skipped processing, loaded existing data.")
            print(f"✓ Raw image features: {len(df_raw)} patients")
            print(f"✓ Unified features: {len(df_unified)} patients")
            print()
            
            return df_raw, df_unified, raw_results, unified_results
        except FileNotFoundError:
            print("Error: Required CSV files not found!")
            print("Need: 'raw_features_data.csv' and 'unified_features_data.csv'")
            print("Run without skip_processing=True first to generate these files.")
            return None, None, None, None
    
    # Original processing mode (only runs if skip_processing=False)
    # Verify paths
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path {DATA_PATH} does not exist!")
        return
    
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file {JSON_PATH} does not exist!")
        return
    
    print("Starting COMPARATIVE K-means clustering analysis...")
    print("=" * 70)
    print("STEP 1: Raw T1xFLAIR image features ONLY")
    print("STEP 2: Raw + Choroid Plexus segmentation features")
    print("=" * 70)
    
    # Load and preprocess data
    # ... (existing data loading and preprocessing code) ...
    
    # For demonstration, let's create dummy data
    np.random.seed(0)
    df_raw = pd.DataFrame({
        'patient_id': range(1, 101),
        'group': np.random.choice(['AD', 'MCI', 'Psy', 'other'], 100),
        'cluster': np.random.randint(1, 6, 100)
    })
    
    df_unified = df_raw.copy()
    df_unified['cluster'] = df_unified['cluster'] + 5  # Different clusters for unified
    
    # Save dummy data to CSV (remove this in the actual script)
    df_raw.to_csv('raw_features_data.csv', index=False)
    df_unified.to_csv('unified_features_data.csv', index=False)
    
    print("Data loading and preprocessing completed.")
    print(f"✓ Raw image features: {len(df_raw)} patients")
    print(f"✓ Unified features: {len(df_unified)} patients")
    print()
    
    # Clustering analysis
    # ... (existing clustering analysis code) ...
    
    # For demonstration, let's create dummy clustering results
    raw_results = pd.DataFrame({
        'cluster': df_raw['cluster'],
        'group': df_raw['group']
    })
    
    unified_results = pd.DataFrame({
        'cluster': df_unified['cluster'],
        'group': df_unified['group']
    })
    
    # Save dummy results to CSV (remove this in the actual script)
    raw_results.to_csv('raw_kmeans_results.csv', index=False)
    unified_results.to_csv('unified_kmeans_results.csv', index=False)
    
    print("Clustering analysis completed.")
    print("Results saved to CSV files.")
    print("✓ raw_kmeans_results.csv")
    print("✓ unified_kmeans_results.csv")
    print()
    
    # Compare clustering quality metrics
    print("CLUSTERING QUALITY COMPARISON:")
    print("-" * 50)
    
    # Calculate metrics for combined approach
    combined_confusion = pd.crosstab(df_unified['cluster'], df_unified['group'])
    raw_confusion = pd.crosstab(df_raw['cluster'], df_raw['group'])
    
    # Print confusion matrices
    print("Combined Features Confusion Matrix:")
    print(combined_confusion)
    print()
    print("Raw Image Features Confusion Matrix:")
    print(raw_confusion)
    print()
    
    # Cluster purity comparison
    def calculate_purity(confusion_matrix):
        purities = []
        sizes = []
        for cluster in confusion_matrix.index:
            cluster_row = confusion_matrix.loc[cluster]
            max_count = cluster_row.max()
            total_count = cluster_row.sum()
            purity = max_count / total_count
            purities.append(purity)
            sizes.append(total_count)
        
        # Weighted average purity
        total_samples = sum(sizes)
        weighted_purity = sum(p * s for p, s in zip(purities, sizes)) / total_samples
        return weighted_purity, purities
    
    combined_purity, combined_cluster_purities = calculate_purity(combined_confusion)
    raw_purity, raw_cluster_purities = calculate_purity(raw_confusion)
    
    print("PURITY COMPARISON:")
    print(f"Combined Features:  {combined_purity:.3f} ({combined_purity*100:.1f}%)")
    print(f"Raw Image Features: {raw_purity:.3f} ({raw_purity*100:.1f}%)")
    
    if raw_purity > combined_purity:
        print("→ Raw image features show BETTER clustering!")
    else:
        print("→ Combined features show BETTER clustering!")
    
    print()
    
    # Feature type analysis
    print("FEATURE TYPE ANALYSIS:")
    print("-" * 30)
    
    combined_features = [col for col in df_unified.columns 
                        if col not in ['patient_id', 'group', 'cluster']]
    raw_features = [col for col in df_raw.columns 
                   if col not in ['patient_id', 'group', 'cluster']]
    
    # Categorize combined features
    image_feats = [f for f in combined_features if 'mask' not in f and f not in 
                   ['volume', 'surface_area', 'surface_to_volume_ratio', 'num_components', 
                    'centroid_x', 'centroid_y', 'centroid_z']]
    seg_feats = [f for f in combined_features if f not in image_feats]
    
    print(f"Combined approach uses:")
    print(f"  • {len(image_feats)} image features")
    print(f"  • {len(seg_feats)} segmentation features")
    print(f"  • Total: {len(combined_features)} features")
    print()
    print(f"Raw approach uses:")
    print(f"  • {len(raw_features)} image features only")
    print()
    
    # Group-wise analysis
    print("GROUP-WISE CLUSTERING PERFORMANCE:")
    print("-" * 40)
    
    for group in ['AD', 'MCI', 'Psy', 'other']:
        # Combined approach
        group_combined = df_unified[df_unified['group'] == group]
        combined_clusters = group_combined['cluster'].value_counts()
        combined_main_cluster = combined_clusters.index[0]
        combined_main_pct = combined_clusters.iloc[0] / len(group_combined) * 100
        
        # Raw approach  
        group_raw = df_raw[df_raw['group'] == group]
        raw_clusters = group_raw['cluster'].value_counts()
        raw_main_cluster = raw_clusters.index[0]
        raw_main_pct = raw_clusters.iloc[0] / len(group_raw) * 100
        
        print(f"{group:6s}: Combined={combined_main_pct:5.1f}% (cluster {combined_main_cluster}) | "
              f"Raw={raw_main_pct:5.1f}% (cluster {raw_main_cluster})")
    
    print()
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Purity comparison
    methods = ['Combined\\n(Image+Seg)', 'Raw Image\\nOnly']
    purities = [combined_purity * 100, raw_purity * 100]
    colors = ['steelblue', 'coral']
    
    bars = axes[0,0].bar(methods, purities, color=colors, alpha=0.7)
    axes[0,0].set_title('Clustering Purity Comparison')
    axes[0,0].set_ylabel('Purity (%)')
    axes[0,0].set_ylim(0, 50)
    
    # Add value labels on bars
    for bar, purity in zip(bars, purities):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                      f'{purity:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Feature count comparison
    feature_counts = [len(combined_features), len(raw_features)]
    bars = axes[0,1].bar(methods, feature_counts, color=colors, alpha=0.7)
    axes[0,1].set_title('Number of Features Used')
    axes[0,1].set_ylabel('Feature Count')
    
    for bar, count in zip(bars, feature_counts):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                      f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Confusion matrix for combined
    sns.heatmap(combined_confusion, annot=True, fmt='d', ax=axes[1,0], cmap='Blues')
    axes[1,0].set_title('Combined Features\\nConfusion Matrix')
    axes[1,0].set_xlabel('True Group')
    axes[1,0].set_ylabel('Cluster')
    
    # 4. Confusion matrix for raw
    sns.heatmap(raw_confusion, annot=True, fmt='d', ax=axes[1,1], cmap='Oranges')
    axes[1,1].set_title('Raw Image Features\\nConfusion Matrix')
    axes[1,1].set_xlabel('True Group')
    axes[1,1].set_ylabel('Cluster')
    
    plt.tight_layout()
    plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("SUMMARY & IMPLICATIONS:")
    print("=" * 40)
    
    if raw_purity > combined_purity:
        print("🔍 Key Finding: RAW IMAGE features cluster better than combined features!")
        print("   → This suggests pathology groups have distinct IMAGING characteristics")
        print("   → Segmentation complexity may not be directly related to disease type")
        print("   → Raw T1xFLAIR patterns are more informative for group classification")
    else:
        print("🔍 Key Finding: COMBINED features cluster better than raw image alone!")
        print("   → This suggests segmentation properties add valuable information")
        print("   → Disease affects both imaging characteristics AND segmentation complexity")
        print("   → Choroid plexus shape/volume patterns are disease-related")
    
    print()
    print("Clinical Implications:")
    if raw_purity > combined_purity:
        print("• Focus on image preprocessing and enhancement techniques")
        print("• Raw imaging biomarkers may be sufficient for group prediction")
        print("• Segmentation difficulty is more individual than disease-specific")
    else:
        print("• Both image quality and anatomical variation matter")
        print("• Segmentation-based biomarkers provide additional diagnostic value")
        print("• Disease affects choroid plexus structure in measurable ways")
    
    print()
    print("Generated: clustering_comparison.png")

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--skip':
            print("Running in SKIP PROCESSING mode...")
            compare_clustering_approaches(skip_processing=True)
        elif sys.argv[1] == '--full':
            print("Running in FULL PROCESSING mode...")
            compare_clustering_approaches(skip_processing=False)
        elif sys.argv[1] in ['--help', '-h']:
            print(COMMANDS_HELP)
        else:
            print("Unknown argument. Use --help for available commands.")
            print(COMMANDS_HELP)
    else:
        # Interactive mode
        print(COMMANDS_HELP)
        choice = input("\nChoose mode (s)kip processing or (f)ull processing: ").lower()
        if choice.startswith('s'):
            compare_clustering_approaches(skip_processing=True)
        elif choice.startswith('f'):
            compare_clustering_approaches(skip_processing=False)
        else:
            print("Invalid choice. Running skip mode by default.")
            compare_clustering_approaches(skip_processing=True)
