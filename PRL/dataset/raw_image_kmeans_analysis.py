"""
K-means Clustering Analysis on RAW IMAGE FEATURES ONLY
Choroid Plexus Segmentation Dataset

This script clusters patients based ONLY on raw T1xFLAIR image characteristics,
without using any ground truth segmentation information.

Author: Generated for PhD Thesis Analysis
Date: September 2025
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
from scipy import ndimage

def calculate_cluster_accuracy(true_groups, cluster_labels):
    """Calculate cluster purity and accuracy metrics"""
    import pandas as pd
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    
    # Create confusion matrix
    df_temp = pd.DataFrame({'true': true_groups, 'cluster': cluster_labels})
    confusion_matrix = pd.crosstab(df_temp['cluster'], df_temp['true'])
    
    # Calculate purity
    cluster_purities = []
    cluster_sizes = []
    
    for cluster_id in confusion_matrix.index:
        cluster_row = confusion_matrix.loc[cluster_id]
        max_group_count = cluster_row.max()
        total_in_cluster = cluster_row.sum()
        purity = max_group_count / total_in_cluster if total_in_cluster > 0 else 0 # proportion of the cluster that belongs to its dominant group
        
        cluster_purities.append(purity)
        cluster_sizes.append(total_in_cluster)
    
    # Overall purity
    total_samples = sum(cluster_sizes)
    overall_purity = sum(p * s for p, s in zip(cluster_purities, cluster_sizes)) / total_samples # weighted average purity
    
    # Best accuracy using Hungarian algorithm: find the optimal mapping between clusters and true groups that maximizes accuracy
    cost_matrix = confusion_matrix.values.max() - confusion_matrix.values # invert as linear_sum_assignment does minimization
    cluster_indices, group_indices = linear_sum_assignment(cost_matrix) # maximize the number of correctly assigned data points
    
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

def extract_raw_image_features(image_data):
    """Extract features ONLY from raw T1xFLAIR image data (NO segmentation info)"""
    features = {}
    
    # Remove background/zero voxels and normalize intensity
    # Use robust percentile-based normalization to handle outliers
    non_zero_mask = image_data > 0
    if np.sum(non_zero_mask) == 0:
        # Handle case where entire image is zero/background
        image_normalized = image_data
    else:
        # Use 1st and 99th percentile for robust normalization
        p1, p99 = np.percentile(image_data[non_zero_mask], [1, 99])
        if p99 > p1:
            image_normalized = np.clip((image_data - p1) / (p99 - p1), 0, 1)
        else:
            image_normalized = image_data / np.max(image_data) if np.max(image_data) > 0 else image_data
    
    # Basic intensity statistics (on normalized data)
    features['mean_intensity'] = np.mean(image_normalized)
    features['std_intensity'] = np.std(image_normalized)
    features['min_intensity'] = np.min(image_normalized)
    features['max_intensity'] = np.max(image_normalized)
    features['median_intensity'] = np.median(image_normalized)
    features['intensity_range'] = features['max_intensity'] - features['min_intensity']
    
    # Intensity distribution features (on normalized data)
    features['intensity_skewness'] = calculate_skewness(image_normalized.flatten())
    features['intensity_kurtosis'] = calculate_kurtosis(image_normalized.flatten())
    
    # Percentiles (on normalized data)
    features['p25_intensity'] = np.percentile(image_normalized, 25)
    features['p75_intensity'] = np.percentile(image_normalized, 75)
    features['iqr_intensity'] = features['p75_intensity'] - features['p25_intensity']
    
    # Texture features (gradients on normalized data)
    grad_x = ndimage.sobel(image_normalized, axis=0)
    grad_y = ndimage.sobel(image_normalized, axis=1)
    grad_z = ndimage.sobel(image_normalized, axis=2)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    features['gradient_mean'] = np.mean(gradient_magnitude)
    features['gradient_std'] = np.std(gradient_magnitude)
    features['gradient_max'] = np.max(gradient_magnitude)
    
    # Laplacian (measures texture/edges on normalized data)
    laplacian = ndimage.laplace(image_normalized.astype(float))
    features['laplacian_mean'] = np.mean(np.abs(laplacian))
    features['laplacian_std'] = np.std(laplacian)
    features['laplacian_var'] = np.var(laplacian)
    
    # Local variance (texture measure on normalized data)
    # Use small kernel for local variance calculation
    kernel_size = 3
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(image_normalized.astype(float), kernel_size)
    local_variance = uniform_filter(image_normalized**2, kernel_size) - local_mean**2
    features['local_variance_mean'] = np.mean(local_variance)
    features['local_variance_std'] = np.std(local_variance)
    
    # Entropy (randomness measure on normalized data)
    hist, _ = np.histogram(image_normalized, bins=256, density=True)
    hist = hist[hist > 0]  # Remove zeros to avoid log(0)
    features['entropy'] = -np.sum(hist * np.log2(hist))
    
    return features

def calculate_skewness(data):
    """Calculate skewness"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 3)

def calculate_kurtosis(data):
    """Calculate kurtosis"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0
    return np.mean(((data - mean) / std) ** 4) - 3

def process_patients_raw_only(data_path, json_path):
    """Process all patients and extract RAW IMAGE features only"""
    print("Loading patient groups...")
    patient_groups = load_patient_groups(json_path)
    
    all_features = []
    patient_ids = []
    groups = []
    
    print("Processing patients (RAW IMAGE FEATURES ONLY)...")
    for patient_id in tqdm(range(1, 105)):  # 001 to 104
        patient_folder = f"{patient_id:03d}"
        patient_path = os.path.join(data_path, patient_folder)
        
        if not os.path.exists(patient_path):
            continue
            
        # Only need the raw T1xFLAIR image
        image_file = f"{patient_folder}_T1xFLAIR.nii"
        image_path = os.path.join(patient_path, image_file)
        
        # Skip if file doesn't exist
        if not os.path.exists(image_path):
            print(f"Warning: Image file missing for patient {patient_folder}")
            continue
            
        try:
            # Load only the raw image
            image_nii = nib.load(image_path)
            image_data = image_nii.get_fdata()
            
            # Extract raw image features only
            features = extract_raw_image_features(image_data)
            features['patient_id'] = patient_id
            features['group'] = patient_groups.get(str(patient_id), 'unknown')
            # print all features['group']
            print(f"Patient {patient_id} group: {features['group']}")

            all_features.append(features)
            patient_ids.append(patient_id)
            groups.append(features['group'])
            
        except Exception as e:
            print(f"Error processing patient {patient_folder}: {e}")
            continue
    
    df = pd.DataFrame(all_features)
    print(f"Successfully processed {len(df)} patients (RAW FEATURES ONLY)")
    return df

def perform_kmeans_analysis_raw(df, k=4):
    """Perform K-means clustering on raw image features"""
    print(f"Performing K-means clustering on RAW IMAGE FEATURES with k={k}...")
    
    # Prepare features for clustering (exclude patient_id and group)
    feature_cols = [col for col in df.columns 
                   if col not in ['patient_id', 'group']]
    X = df[feature_cols].values
    
    print(f"Using {len(feature_cols)} raw image features:")
    for feat in feature_cols:
        print(f"  • {feat}")
    print()
    
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
    
    # Calculate clustering metrics. 
    # evaluate how well-separated and internally cohesive the clusters are, without needing ground truth labels
    silhouette_avg = silhouette_score(X_scaled, cluster_labels) # in -1...1 well-matched to their clusters and poorly matched to neighboring clusters
    # 0 overlapping clusters, and negative values indicate that points might be assigned to the wrong clusters.
    
    # Compare clusters with actual groups
    group_mapping = {'AD': 0, 'MCI': 1, 'Psy': 2, 'other': 3}
    true_labels = [group_mapping.get(group, 3) for group in df['group']]
    # measure how well the clustering results match the true medical group classifications (requires GT): -0.5 ... 1.0
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    # 0: random clustering performance
    # worse-than-random clustering
    
    # Calculate cluster purity and accuracy
    cluster_purity, cluster_accuracy, best_mapping = calculate_cluster_accuracy(df['group'], cluster_labels)
    
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Adjusted Rand Index: {ari_score:.3f}")
    print(f"Cluster Purity: {cluster_purity:.3f} ({cluster_purity*100:.1f}%)")
    print(f"Best Cluster Accuracy: {cluster_accuracy:.3f} ({cluster_accuracy*100:.1f}%)")
    
    # write results and features to csv:
    results_df = pd.DataFrame({
        'silhouette_score': [silhouette_avg],
        'ari_score': [ari_score],
        'cluster_purity': [cluster_purity],
        'cluster_accuracy': [cluster_accuracy]
    })
    results_df.to_csv('clustering_results_raw.csv', index=False)
    df.to_csv('features_data_raw.csv', index=False)
    

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

def visualize_results_raw(df, clustering_results, show_accuracy_borders=True):
    """Create visualizations for raw image clustering
    
    Args:
        df: DataFrame with clustering results
        clustering_results: Dictionary with clustering metrics and results
        show_accuracy_borders: If True, add colored borders to show classification accuracy
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Group distribution
    df['group'].value_counts().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Distribution of Pathology Groups')
    axes[0,0].set_xlabel('Group')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Mean intensity by group
    sns.boxplot(data=df, x='group', y='mean_intensity', ax=axes[0,1])
    axes[0,1].set_title('Mean Intensity by Group (Raw T1xFLAIR)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. K-means clusters (PCA projection)
    X_scaled = clustering_results['X_scaled']
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Use distinct colors
    cluster_colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12']
    unique_clusters = sorted(df['cluster'].unique())
    
    # Get optimal mapping for accuracy borders
    best_mapping = clustering_results['best_mapping']
    
    for i, cluster in enumerate(unique_clusters):
        mask = df['cluster'] == cluster
        
        if show_accuracy_borders:
            # Get the optimal group for this cluster
            optimal_group = best_mapping.get(cluster, None)
            
            # Separate correctly and incorrectly classified points
            cluster_data = df[mask]
            correct_mask = cluster_data['group'] == optimal_group
            incorrect_mask = ~correct_mask
            
            # Plot correctly classified points with green border
            if correct_mask.any():
                correct_indices = cluster_data[correct_mask].index
                correct_pca_indices = [idx for idx in range(len(df)) if df.index[idx] in correct_indices]
                axes[1,0].scatter(X_pca[correct_pca_indices, 0], X_pca[correct_pca_indices, 1], 
                                c=cluster_colors[i], s=50, alpha=0.7,
                                edgecolors='green', linewidths=1.5)
            
            # Plot incorrectly classified points without border (clean look)
            if incorrect_mask.any():
                incorrect_indices = cluster_data[incorrect_mask].index
                incorrect_pca_indices = [idx for idx in range(len(df)) if df.index[idx] in incorrect_indices]
                axes[1,0].scatter(X_pca[incorrect_pca_indices, 0], X_pca[incorrect_pca_indices, 1], 
                                c=cluster_colors[i], s=50, alpha=0.7)
            
            # Add cluster label (only once per cluster)
            axes[1,0].scatter([], [], c=cluster_colors[i], label=f'Cluster {cluster}', 
                            alpha=0.7, s=50)
        else:
            # Original plotting without borders
            axes[1,0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                             c=cluster_colors[i], label=f'Cluster {cluster}', 
                             alpha=0.7, s=50)
    
    title = 'K-means Clusters (PCA projection)\nRAW IMAGE FEATURES ONLY'
    if show_accuracy_borders:
        title += '\nGreen border: Correctly classified'
    
    axes[1,0].set_title(title)
    axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[1,0].legend(title='Clusters', loc='upper right')
    
    # 4. Cluster vs Group comparison
    cluster_group_df = pd.crosstab(df['group'], df['cluster'])
    cluster_group_pct = pd.crosstab(df['group'], df['cluster'], normalize='columns') * 100
    
    annot_text = []
    for i in range(cluster_group_df.shape[0]):
        row = []
        for j in range(cluster_group_df.shape[1]):
            count = cluster_group_df.iloc[i, j]
            pct = cluster_group_pct.iloc[i, j]
            row.append(f'{count}\n({pct:.1f}%)')
        annot_text.append(row)
    
    sns.heatmap(cluster_group_df, annot=annot_text, fmt='', ax=axes[1,1], cmap='Blues')
    axes[1,1].set_title('Cluster vs Pathology Group\n(RAW FEATURES - Count and % of cluster)')
    axes[1,1].set_xlabel('Cluster')
    axes[1,1].set_ylabel('Pathology Group')
    
    plt.tight_layout()
    plt.savefig('kmeans_raw_features_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance plot - top features
    plt.figure(figsize=(14, 10))
    cluster_centers = clustering_results['kmeans'].cluster_centers_
    feature_names = clustering_results['feature_cols']
    
    for i in range(len(cluster_centers)):
        plt.subplot(2, 2, i+1)
        bars = plt.bar(range(len(feature_names)), cluster_centers[i])
        plt.title(f'Cluster {i} Raw Feature Profile')
        plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
        plt.ylabel('Standardized Value')
        
        # Color bars based on positive/negative values
        for j, bar in enumerate(bars):
            if cluster_centers[i][j] > 0:
                bar.set_color('steelblue')
            else:
                bar.set_color('coral')
    
    plt.tight_layout()
    plt.savefig('cluster_raw_feature_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_report_raw(df, clustering_results):
    """Generate summary report for raw image clustering"""
    report = []
    report.append("K-MEANS CLUSTERING - RAW IMAGE FEATURES ONLY")
    report.append("=" * 60)
    report.append("")
    report.append("ANALYSIS TYPE: Raw T1xFLAIR image characteristics")
    report.append("NO SEGMENTATION INFORMATION USED")
    report.append("")
    
    # Dataset overview
    report.append(f"Total patients: {len(df)}")
    report.append("Group distribution:")
    for group, count in df['group'].value_counts().items():
        report.append(f"  {group}: {count} patients")
    report.append("")
    
    # Feature summary
    feature_count = len(clustering_results['feature_cols'])
    report.append(f"Features used: {feature_count} raw image characteristics")
    report.append("Feature types:")
    report.append("  • Intensity statistics (mean, std, percentiles)")
    report.append("  • Texture measures (gradients, Laplacian, local variance)")
    report.append("  • Distribution properties (skewness, kurtosis, entropy)")
    report.append("")
    
    # Clustering results
    report.append("Clustering Results:")
    report.append(f"  Silhouette Score: {clustering_results['silhouette_score']:.3f}")
    report.append(f"  Adjusted Rand Index: {clustering_results['ari_score']:.3f}")
    report.append(f"  Cluster Purity: {clustering_results['cluster_purity']:.3f} ({clustering_results['cluster_purity']*100:.1f}%)")
    report.append(f"  Best Cluster Accuracy: {clustering_results['cluster_accuracy']:.3f} ({clustering_results['cluster_accuracy']*100:.1f}%)")
    report.append("")
    
    # Best mapping
    report.append("Optimal Cluster-to-Group Mapping:")
    for cluster_id, group_name in clustering_results['best_mapping'].items():
        report.append(f"  Cluster {cluster_id} → {group_name}")
    report.append("")
    
    # Interpretation
    if clustering_results['silhouette_score'] > 0.5:
        complexity = "HIGH - Raw images show distinct patterns between groups"
    elif clustering_results['silhouette_score'] > 0.25:
        complexity = "MODERATE - Some imaging patterns exist between groups"
    else:
        complexity = "LOW - Limited distinct imaging patterns between groups"
    
    report.append(f"Imaging Pattern Complexity: {complexity}")
    report.append("")
    
    # Group statistics
    report.append("Group Statistics (mean ± std):")
    for group in ['AD', 'MCI', 'Psy', 'other']:
        group_data = df[df['group'] == group]
        if len(group_data) > 0:
            intensity_mean = group_data['mean_intensity'].mean()
            intensity_std = group_data['mean_intensity'].std()
            entropy_mean = group_data['entropy'].mean()
            report.append(f"  {group}: Intensity={intensity_mean:.0f}±{intensity_std:.0f}, Entropy={entropy_mean:.2f}")
    
    report.append("")
    report.append("COMPARISON WITH SEGMENTATION-BASED CLUSTERING:")
    report.append("• This analysis shows if pathology groups have distinct IMAGING patterns")
    report.append("• Previous analysis used segmentation features (volume, shape)")
    report.append("• Compare results to see if disease affects raw images vs segmentation complexity")
    report.append("")
    report.append("Files generated:")
    report.append("  - kmeans_raw_features_results.png")
    report.append("  - cluster_raw_feature_profiles.png")
    report.append("  - features_data_raw.csv")
    
    # Save report
    with open('kmeans_raw_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Print report
    for line in report:
        print(line)

def main():
    """Main function for raw image clustering"""
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
    
    print("Starting K-means clustering analysis on RAW IMAGE FEATURES...")
    print("=" * 70)
    print("NOTE: This analysis uses ONLY raw T1xFLAIR image characteristics")
    print("NO segmentation information is used!")
    print("=" * 70)
    
    # 1. Process patients (raw features only)
    df = process_patients_raw_only(DATA_PATH, JSON_PATH)
    
    if len(df) == 0:
        print("Error: No data processed!")
        return
    
    # 2. K-means analysis on raw features
    clustering_results = perform_kmeans_analysis_raw(df, k=4)
    
    # 3. Visualize results
    visualize_results_raw(df, clustering_results, show_accuracy_borders=True)
    
    # 4. Generate report
    generate_summary_report_raw(df, clustering_results)
    
    # 5. Save data
    df.to_csv('features_data_raw.csv', index=False)
    
    print("\nRAW IMAGE CLUSTERING ANALYSIS complete!")
    
    return df, clustering_results

if __name__ == "__main__":
    df, results = main()
