"""
Simplified K-means Clustering Analysis for Choroid Plexus Segmentation Dataset

This script focuses on K-means clustering (k=4) to analyze dataset complexity
for choroid plexus segmentation across pathology groups.
"""

import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def calculate_cluster_accuracy(true_groups, cluster_labels):
    """
    Calculate cluster purity and accuracy metrics
    
    Args:
        true_groups: List of true group labels (e.g., ['AD', 'MCI', 'Psy', 'other'])
        cluster_labels: List of cluster assignments (e.g., [0, 1, 2, 3])
    
    Returns:
        purity: Overall purity of clustering
        accuracy: Best possible accuracy with optimal mapping
        best_mapping: Dictionary mapping clusters to groups for best accuracy
    """
    import pandas as pd
    from collections import Counter
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    
    # Create confusion matrix
    df_temp = pd.DataFrame({'true': true_groups, 'cluster': cluster_labels})
    confusion_matrix = pd.crosstab(df_temp['cluster'], df_temp['true'])
    
    # Calculate purity (weighted average of max group in each cluster)
    cluster_purities = []
    cluster_sizes = []
    
    for cluster_id in confusion_matrix.index:
        cluster_row = confusion_matrix.loc[cluster_id]
        max_group_count = cluster_row.max()
        total_in_cluster = cluster_row.sum()
        purity = max_group_count / total_in_cluster if total_in_cluster > 0 else 0
        
        cluster_purities.append(purity)
        cluster_sizes.append(total_in_cluster)
    
    # Overall purity (weighted by cluster size)
    total_samples = sum(cluster_sizes)
    overall_purity = sum(p * s for p, s in zip(cluster_purities, cluster_sizes)) / total_samples
    
    # Calculate best accuracy using Hungarian algorithm (optimal assignment)
    cost_matrix = confusion_matrix.values.max() - confusion_matrix.values  # Convert to cost
    cluster_indices, group_indices = linear_sum_assignment(cost_matrix)
    
    # Get the optimal mapping
    best_mapping = {}
    total_correct = 0
    
    group_names = list(confusion_matrix.columns)
    
    for cluster_idx, group_idx in zip(cluster_indices, group_indices):
        cluster_id = confusion_matrix.index[cluster_idx]
        group_name = group_names[group_idx]
        best_mapping[cluster_id] = group_name
        total_correct += confusion_matrix.iloc[cluster_idx, group_idx]
    
    best_accuracy = total_correct / total_samples
    
    return overall_purity, best_accuracy, best_mapping

def load_patient_groups(json_path):
    """Load patient group mappings from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['index_order_all']

def extract_basic_features(image_data, mask_data):
    """Extract basic features from image and mask data"""
    features = {}
    
    # Image features
    features['mean_intensity'] = np.mean(image_data)
    features['std_intensity'] = np.std(image_data)
    features['min_intensity'] = np.min(image_data)
    features['max_intensity'] = np.max(image_data)
    features['intensity_range'] = features['max_intensity'] - features['min_intensity']
    
    # Mask features (choroid plexus)
    mask_binary = mask_data > 0
    features['volume'] = np.sum(mask_binary)
    
    if features['volume'] > 0:
        # Mean intensity within mask
        features['mask_mean_intensity'] = np.mean(image_data[mask_binary])
        features['mask_std_intensity'] = np.std(image_data[mask_binary])
        
        # Shape features
        # Approximate surface area using edge detection
        from scipy import ndimage
        edges = ndimage.sobel(mask_data.astype(float))
        features['surface_area'] = np.sum(edges > 0)
        features['surface_to_volume_ratio'] = features['surface_area'] / features['volume']
        
        # Connected components
        labeled, num_components = ndimage.label(mask_binary)
        features['num_components'] = num_components
        
        # Center of mass
        center = ndimage.center_of_mass(mask_binary)
        features['centroid_x'] = center[0] if not np.isnan(center[0]) else 0
        features['centroid_y'] = center[1] if not np.isnan(center[1]) else 0
        features['centroid_z'] = center[2] if not np.isnan(center[2]) else 0
    else:
        # Handle empty masks
        features.update({
            'mask_mean_intensity': 0,
            'mask_std_intensity': 0,
            'surface_area': 0,
            'surface_to_volume_ratio': 0,
            'num_components': 0,
            'centroid_x': 0, 'centroid_y': 0, 'centroid_z': 0
        })
    
    return features

def process_patients(data_path, json_path):
    """Process all patients and extract features"""
    print("Loading patient groups...")
    patient_groups = load_patient_groups(json_path)
    
    all_features = []
    patient_ids = []
    groups = []
    
    print("Processing patients...")
    for patient_id in tqdm(range(1, 105)):  # 001 to 104
        patient_folder = f"{patient_id:03d}"
        patient_path = os.path.join(data_path, patient_folder)
        
        if not os.path.exists(patient_path):
            continue
            
        # File paths
        image_file = f"{patient_folder}_T1xFLAIR.nii"
        mask_file = f"{patient_folder}_ChP_mask_T1xFLAIR_manual_seg.nii"
        
        image_path = os.path.join(patient_path, image_file)
        mask_path = os.path.join(patient_path, mask_file)
        
        # Skip if files don't exist
        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            print(f"Warning: Files missing for patient {patient_folder}")
            continue
            
        try:
            # Load images
            image_nii = nib.load(image_path)
            mask_nii = nib.load(mask_path)
            
            image_data = image_nii.get_fdata()
            mask_data = mask_nii.get_fdata()
            
            # Extract features
            features = extract_basic_features(image_data, mask_data)
            features['patient_id'] = patient_id
            features['group'] = patient_groups.get(str(patient_id), 'unknown')
            
            all_features.append(features)
            patient_ids.append(patient_id)
            groups.append(features['group'])
            
        except Exception as e:
            print(f"Error processing patient {patient_folder}: {e}")
            continue
    
    df = pd.DataFrame(all_features)
    print(f"Successfully processed {len(df)} patients")
    return df

def perform_kmeans_analysis(df, k=4):
    """Perform K-means clustering analysis"""
    print(f"Performing K-means clustering with k={k}...")
    
    # Prepare features for clustering
    feature_cols = [col for col in df.columns 
                   if col not in ['patient_id', 'group']]
    X = df[feature_cols].values
    
    # Handle NaN values
    X = np.nan_to_num(X)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to DataFrame
    df['cluster'] = cluster_labels
    
    # Calculate clustering metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    
    # Compare clusters with actual groups
    group_mapping = {'AD': 0, 'MCI': 1, 'Psy': 2, 'other': 3}
    true_labels = [group_mapping.get(group, 3) for group in df['group']]
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    
    # Calculate cluster purity and accuracy
    cluster_purity, cluster_accuracy, best_mapping = calculate_cluster_accuracy(df['group'], cluster_labels)
    
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Adjusted Rand Index: {ari_score:.3f}")
    print(f"Cluster Purity: {cluster_purity:.3f} ({cluster_purity*100:.1f}%)")
    print(f"Best Cluster Accuracy: {cluster_accuracy:.3f} ({cluster_accuracy*100:.1f}%)")
    
    return {
        'silhouette_score': silhouette_avg,
        'ari_score': ari_score,
        'cluster_purity': cluster_purity,
        'cluster_accuracy': cluster_accuracy,
        'best_mapping': best_mapping,
        'cluster_labels': cluster_labels,
        'X_scaled': X_scaled,
        'scaler': scaler,
        'kmeans': kmeans,
        'feature_cols': feature_cols
    }

def visualize_results(df, clustering_results):
    """Create visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Group distribution
    df['group'].value_counts().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Distribution of Pathology Groups')
    axes[0,0].set_xlabel('Group')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Volume by group
    sns.boxplot(data=df, x='group', y='volume', ax=axes[0,1])
    axes[0,1].set_title('Choroid Plexus Volume by Group')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. K-means clusters (PCA projection)
    X_scaled = clustering_results['X_scaled']
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Use discrete colors for categorical clusters - highly distinct colors
    cluster_colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12']  # Red, Green, Blue, Orange
    unique_clusters = sorted(df['cluster'].unique())
    
    for i, cluster in enumerate(unique_clusters):
        mask = df['cluster'] == cluster
        axes[1,0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                         c=cluster_colors[i], label=f'Cluster {cluster}', 
                         alpha=0.7, s=50)
    
    axes[1,0].set_title('K-means Clusters (PCA projection)')
    axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[1,0].legend(title='Clusters', loc='upper right')
    
    # 4. Cluster vs Group comparison with percentages
    cluster_group_df = pd.crosstab(df['group'], df['cluster'])
    cluster_group_pct = pd.crosstab(df['group'], df['cluster'], normalize='columns') * 100
    
    # Create annotation combining counts and percentages
    annot_text = []
    for i in range(cluster_group_df.shape[0]):
        row = []
        for j in range(cluster_group_df.shape[1]):
            count = cluster_group_df.iloc[i, j]
            pct = cluster_group_pct.iloc[i, j]
            row.append(f'{count}\n({pct:.1f}%)')
        annot_text.append(row)
    
    sns.heatmap(cluster_group_df, annot=annot_text, fmt='', ax=axes[1,1], cmap='Blues')
    axes[1,1].set_title('Cluster vs Pathology Group\n(Count and % of cluster)')
    axes[1,1].set_xlabel('Cluster')
    axes[1,1].set_ylabel('Pathology Group')
    
    plt.tight_layout()
    plt.savefig('kmeans_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance (cluster centers)
    plt.figure(figsize=(12, 8))
    cluster_centers = clustering_results['kmeans'].cluster_centers_
    feature_names = clustering_results['feature_cols']
    
    for i in range(len(cluster_centers)):
        plt.subplot(2, 2, i+1)
        plt.bar(range(len(feature_names)), cluster_centers[i])
        plt.title(f'Cluster {i} Feature Profile')
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.ylabel('Standardized Value')
    
    plt.tight_layout()
    plt.savefig('cluster_feature_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report(df, clustering_results):
    """Generate a summary report"""
    report = []
    report.append("K-MEANS CLUSTERING ANALYSIS SUMMARY")
    report.append("=" * 50)
    report.append("")
    
    # Dataset overview
    report.append(f"Total patients: {len(df)}")
    report.append("Group distribution:")
    for group, count in df['group'].value_counts().items():
        report.append(f"  {group}: {count} patients")
    report.append("")
    
    # Clustering results
    report.append("Clustering Results:")
    report.append(f"  Silhouette Score: {clustering_results['silhouette_score']:.3f}")
    report.append(f"  Adjusted Rand Index: {clustering_results['ari_score']:.3f}")
    report.append(f"  Cluster Purity: {clustering_results['cluster_purity']:.3f} ({clustering_results['cluster_purity']*100:.1f}%)")
    report.append(f"  Best Cluster Accuracy: {clustering_results['cluster_accuracy']:.3f} ({clustering_results['cluster_accuracy']*100:.1f}%)")
    report.append("")
    
    # Best cluster-to-group mapping
    report.append("Optimal Cluster-to-Group Mapping:")
    for cluster_id, group_name in clustering_results['best_mapping'].items():
        report.append(f"  Cluster {cluster_id} → {group_name}")
    report.append("")
    
    # Interpretation
    if clustering_results['silhouette_score'] > 0.5:
        complexity = "HIGH - Well-separated clusters suggest distinct complexity patterns"
    elif clustering_results['silhouette_score'] > 0.25:
        complexity = "MODERATE - Some clustering structure exists"
    else:
        complexity = "LOW - Limited distinct patterns in the data"
    
    report.append(f"Dataset Complexity: {complexity}")
    report.append("")
    
    # Group statistics
    report.append("Group Statistics (mean ± std):")
    for group in ['AD', 'MCI', 'Psy', 'other']:
        group_data = df[df['group'] == group]
        if len(group_data) > 0:
            vol_mean = group_data['volume'].mean()
            vol_std = group_data['volume'].std()
            int_mean = group_data['mean_intensity'].mean()
            report.append(f"  {group}: Volume={vol_mean:.0f}±{vol_std:.0f}, Intensity={int_mean:.2f}")
    
    report.append("")
    report.append("Files generated:")
    report.append("  - kmeans_analysis_results.png")
    report.append("  - cluster_feature_profiles.png")
    report.append("  - features_data.csv")
    
    # Save report
    with open('kmeans_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Print report
    for line in report:
        print(line)

def main():
    """Main function"""
    # Configuration
    DATA_PATH = "/mnt/pazienti"
    JSON_PATH = "/mnt/pazienti/patients.json"
    
    # Verify paths
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path {DATA_PATH} does not exist!")
        return
    
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file {JSON_PATH} does not exist!")
        return
    
    print("Starting K-means clustering analysis...")
    print("=" * 50)
    
    # 1. Process patients
    df = process_patients(DATA_PATH, JSON_PATH)
    
    if len(df) == 0:
        print("Error: No data processed!")
        return
    
    # 2. K-means analysis
    clustering_results = perform_kmeans_analysis(df, k=4)
    
    # 3. Visualize results
    visualize_results(df, clustering_results)
    
    # 4. Generate report
    generate_summary_report(df, clustering_results)
    
    # 5. Save data
    df.to_csv('features_data.csv', index=False)
    
    print("\nAnalysis complete!")
    
    return df, clustering_results

if __name__ == "__main__":
    df, results = main()
