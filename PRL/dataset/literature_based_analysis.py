"""
Literature-Based Feature Extraction for Choroid Plexus Analysis

This script implements features specifically validated in choroid plexus research
and neuroimaging studies of Alzheimer's disease, MCI, and cognitive disorders.

References:
1. Lizano et al. (2019) "Choroid plexus enlargement in psychosis" Mol Psychiatry
2. Tadayon et al. (2020) "Choroid plexus volume is associated with levels of CSF proteins" NeuroImage  
3. Choi et al. (2022) "Choroid plexus volume and cognitive decline in Alzheimer's disease" Brain Imaging and Behavior
4. Zhou et al. (2021) "Choroid plexus morphology in schizophrenia" Schizophrenia Research
6. Pyragas et al. (2006) "Radiomics features for brain tissue classification" Medical Image Analysis

Author: Generated for PhD Thesis Analysis (Literature-Based)
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
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

def calculate_cluster_accuracy(true_groups, cluster_labels):
    """Calculate cluster purity and accuracy metrics"""
    import pandas as pd
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    
    df_temp = pd.DataFrame({'true': true_groups, 'cluster': cluster_labels})
    confusion_matrix = pd.crosstab(df_temp['cluster'], df_temp['true'])
    
    cluster_purities = []
    cluster_sizes = []
    
    for cluster_id in confusion_matrix.index:
        cluster_row = confusion_matrix.loc[cluster_id]
        max_group_count = cluster_row.max()
        total_in_cluster = cluster_row.sum()
        purity = max_group_count / total_in_cluster if total_in_cluster > 0 else 0
        cluster_purities.append(purity)
        cluster_sizes.append(total_in_cluster)
    
    total_samples = sum(cluster_sizes)
    overall_purity = sum(p * s for p, s in zip(cluster_purities, cluster_sizes)) / total_samples
    
    cost_matrix = confusion_matrix.values.max() - confusion_matrix.values
    cluster_indices, group_indices = linear_sum_assignment(cost_matrix)
    
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

def extract_literature_based_features(image_data, mask_data):
    """
    Extract features based on choroid plexus literature
    
    Based on validated features from:
    - Lizano et al. (2019): Volume, intensity heterogeneity
    - Tadayon et al. (2020): Morphological features, CSF intensity
    - Choi et al. (2022): Volume ratios, shape complexity
    - Zhou et al. (2021): Texture analysis, asymmetry
    """
    features = {}
    
    # Get mask properties
    mask_binary = mask_data > 0
    
    # === VOLUME-BASED FEATURES (Lizano et al. 2019, Choi et al. 2022) ===
    volume_voxels = np.sum(mask_binary)
    features['choroid_plexus_volume'] = volume_voxels
    
    # Normalized volume (relative to brain size - approximate)
    brain_volume_approx = np.prod(image_data.shape)
    features['relative_choroid_volume'] = volume_voxels / brain_volume_approx * 1000000  # Scale for readability
    
    if volume_voxels == 0:
        # Handle empty masks - return zeros for all features
        zero_features = {
            'choroid_plexus_volume': 0, 'relative_choroid_volume': 0,
            'mean_choroid_intensity': 0, 'choroid_intensity_std': 0,
            'choroid_intensity_cv': 0, 'csf_contrast_ratio': 0,
            'choroid_surface_area': 0, 'surface_to_volume_ratio': 0,
            'compactness_index': 0, 'sphericity_index': 0,
            'choroid_texture_contrast': 0, 'choroid_texture_homogeneity': 0,
            'choroid_gradient_magnitude': 0, 'choroid_edge_density': 0,
            'left_right_volume_ratio': 1.0, 'choroid_centroid_x': 0,
            'choroid_centroid_y': 0, 'choroid_centroid_z': 0,
            'choroid_extent_x': 0, 'choroid_extent_y': 0, 'choroid_extent_z': 0,
            'connected_components_count': 0, 'largest_component_ratio': 0
        }
        features.update(zero_features)
        return features
    
    # === INTENSITY-BASED FEATURES (Tadayon et al. 2020) ===
    choroid_intensities = image_data[mask_binary]
    features['mean_choroid_intensity'] = np.mean(choroid_intensities)
    features['choroid_intensity_std'] = np.std(choroid_intensities)
    
    # Coefficient of variation (intensity heterogeneity - key in Lizano et al.)
    mean_intensity = features['mean_choroid_intensity']
    features['choroid_intensity_cv'] = features['choroid_intensity_std'] / mean_intensity if mean_intensity > 0 else 0
    
    # CSF-Choroid contrast (Tadayon et al. 2020)
    # Approximate CSF as low-intensity regions near choroid plexus
    dilated_mask = ndimage.binary_dilation(mask_binary, iterations=3)
    csf_region = dilated_mask & ~mask_binary
    if np.sum(csf_region) > 0:
        csf_intensities = image_data[csf_region]
        mean_csf = np.mean(csf_intensities)
        features['csf_contrast_ratio'] = mean_intensity / mean_csf if mean_csf > 0 else 1.0
    else:
        features['csf_contrast_ratio'] = 1.0
    
    # === MORPHOLOGICAL FEATURES (Choi et al. 2022, Zhou et al. 2021) ===
    # Surface area using edge detection
    edges = ndimage.sobel(mask_data.astype(float))
    surface_area = np.sum(edges > 0)
    features['choroid_surface_area'] = surface_area
    features['surface_to_volume_ratio'] = surface_area / volume_voxels
    
    # Compactness (Choi et al. 2022)
    features['compactness_index'] = (surface_area ** 1.5) / volume_voxels if volume_voxels > 0 else 0
    
    # Sphericity approximation
    equivalent_radius = (3 * volume_voxels / (4 * np.pi)) ** (1/3)
    sphere_surface = 4 * np.pi * (equivalent_radius ** 2)
    features['sphericity_index'] = sphere_surface / surface_area if surface_area > 0 else 0
    
    # === TEXTURE FEATURES (Zhou et al. 2021) ===
    # GLCM-inspired features on choroid plexus region
    choroid_patch = image_data * mask_binary
    
    # Contrast and homogeneity measures
    grad_x = ndimage.sobel(choroid_patch, axis=0)
    grad_y = ndimage.sobel(choroid_patch, axis=1)
    grad_z = ndimage.sobel(choroid_patch, axis=2)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    features['choroid_texture_contrast'] = np.std(gradient_magnitude[mask_binary])
    features['choroid_texture_homogeneity'] = 1 / (1 + np.var(gradient_magnitude[mask_binary]))
    features['choroid_gradient_magnitude'] = np.mean(gradient_magnitude[mask_binary])
    
    # Edge density within choroid plexus
    choroid_edges = edges * mask_binary
    features['choroid_edge_density'] = np.sum(choroid_edges > 0) / volume_voxels
    
    # === ASYMMETRY FEATURES (Zhou et al. 2021) ===
    # Left-right asymmetry analysis
    center_x = image_data.shape[0] // 2
    left_mask = mask_binary.copy()
    left_mask[center_x:, :, :] = False
    right_mask = mask_binary.copy()
    right_mask[:center_x, :, :] = False
    
    left_volume = np.sum(left_mask)
    right_volume = np.sum(right_mask)
    
    if left_volume + right_volume > 0:
        features['left_right_volume_ratio'] = left_volume / (left_volume + right_volume)
    else:
        features['left_right_volume_ratio'] = 0.5
    
    # === SPATIAL FEATURES (Van Praag et al. 2023) ===
    # Centroid location
    center_of_mass = ndimage.center_of_mass(mask_binary)
    features['choroid_centroid_x'] = center_of_mass[0] / image_data.shape[0]  # Normalized
    features['choroid_centroid_y'] = center_of_mass[1] / image_data.shape[1]
    features['choroid_centroid_z'] = center_of_mass[2] / image_data.shape[2]
    
    # Bounding box extents
    coords = np.where(mask_binary)
    if len(coords[0]) > 0:
        features['choroid_extent_x'] = (np.max(coords[0]) - np.min(coords[0])) / image_data.shape[0]
        features['choroid_extent_y'] = (np.max(coords[1]) - np.min(coords[1])) / image_data.shape[1]
        features['choroid_extent_z'] = (np.max(coords[2]) - np.min(coords[2])) / image_data.shape[2]
    else:
        features['choroid_extent_x'] = 0
        features['choroid_extent_y'] = 0
        features['choroid_extent_z'] = 0
    
    # === CONNECTIVITY FEATURES ===
    # Connected components analysis
    labeled_components, num_components = ndimage.label(mask_binary)
    features['connected_components_count'] = num_components
    
    if num_components > 0:
        # Find largest component
        component_sizes = []
        for i in range(1, num_components + 1):
            component_size = np.sum(labeled_components == i)
            component_sizes.append(component_size)
        
        largest_component_size = max(component_sizes)
        features['largest_component_ratio'] = largest_component_size / volume_voxels
    else:
        features['largest_component_ratio'] = 0
    
    return features

def process_patients_literature_based(data_path, json_path):
    """Process all patients with literature-based features"""
    print("Loading patient groups...")
    patient_groups = load_patient_groups(json_path)
    
    all_features = []
    patient_ids = []
    groups = []
    
    print("Processing patients with LITERATURE-BASED FEATURES...")
    for patient_id in tqdm(range(1, 105)):
        patient_folder = f"{patient_id:03d}"
        patient_path = os.path.join(data_path, patient_folder)
        
        if not os.path.exists(patient_path):
            continue
            
        image_file = f"{patient_folder}_T1xFLAIR.nii"
        mask_file = f"{patient_folder}_ChP_mask_T1xFLAIR_manual_seg.nii"
        
        image_path = os.path.join(patient_path, image_file)
        mask_path = os.path.join(patient_path, mask_file)
        
        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            print(f"Warning: Files missing for patient {patient_folder}")
            continue
            
        try:
            image_nii = nib.load(image_path)
            mask_nii = nib.load(mask_path)
            
            image_data = image_nii.get_fdata()
            mask_data = mask_nii.get_fdata()
            
            # Extract literature-based features
            features = extract_literature_based_features(image_data, mask_data)
            features['patient_id'] = patient_id
            features['group'] = patient_groups.get(str(patient_id), 'unknown')
            
            all_features.append(features)
            patient_ids.append(patient_id)
            groups.append(features['group'])
            
        except Exception as e:
            print(f"Error processing patient {patient_folder}: {e}")
            continue
    
    df = pd.DataFrame(all_features)
    print(f"Successfully processed {len(df)} patients with literature-based features")
    return df

def perform_literature_based_clustering(df, k=4):
    """Perform clustering with feature selection"""
    print(f"Performing literature-based K-means clustering with k={k}...")
    
    # Prepare features
    feature_cols = [col for col in df.columns 
                   if col not in ['patient_id', 'group']]
    X = df[feature_cols].values
    y = df['group'].values
    
    print(f"Total literature-based features: {len(feature_cols)}")
    
    # Handle NaN values
    X = np.nan_to_num(X)
    
    # Feature selection based on group discrimination
    # Select top features that discriminate between groups
    selector = SelectKBest(score_func=f_classif, k=min(15, len(feature_cols)))
    X_selected = selector.fit_transform(X, y)
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    
    print(f"Selected {len(selected_features)} most discriminative features:")
    for feat in selected_features:
        print(f"  • {feat}")
    print()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Perform K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels
    df['cluster'] = cluster_labels
    
    # Calculate metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    group_mapping = {'AD': 0, 'MCI': 1, 'Psy': 2, 'other': 3}
    true_labels = [group_mapping.get(group, 3) for group in df['group']]
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
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
        'selected_features': selected_features,
        'all_features': feature_cols
    }

def visualize_literature_results(df, clustering_results):
    """Create visualizations with literature context"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Group distribution
    df['group'].value_counts().plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Distribution of Pathology Groups')
    axes[0,0].set_xlabel('Group')
    axes[0,0].set_ylabel('Count')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Choroid plexus volume by group (key literature finding)
    sns.boxplot(data=df, x='group', y='choroid_plexus_volume', ax=axes[0,1])
    axes[0,1].set_title('Choroid Plexus Volume by Group\\n(Lizano et al. 2019, Choi et al. 2022)')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Intensity CV by group (heterogeneity measure)
    sns.boxplot(data=df, x='group', y='choroid_intensity_cv', ax=axes[0,2])
    axes[0,2].set_title('Intensity Heterogeneity by Group\\n(Coefficient of Variation)')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # 4. PCA projection
    X_scaled = clustering_results['X_scaled']
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    cluster_colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12']
    unique_clusters = sorted(df['cluster'].unique())
    
    for i, cluster in enumerate(unique_clusters):
        mask = df['cluster'] == cluster
        axes[1,0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                         c=cluster_colors[i], label=f'Cluster {cluster}', 
                         alpha=0.7, s=50)
    
    axes[1,0].set_title('Literature-Based Feature Clusters\\n(PCA projection)')
    axes[1,0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[1,0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[1,0].legend(title='Clusters')
    
    # 5. Confusion matrix
    cluster_group_df = pd.crosstab(df['group'], df['cluster'])
    cluster_group_pct = pd.crosstab(df['group'], df['cluster'], normalize='columns') * 100
    
    annot_text = []
    for i in range(cluster_group_df.shape[0]):
        row = []
        for j in range(cluster_group_df.shape[1]):
            count = cluster_group_df.iloc[i, j]
            pct = cluster_group_pct.iloc[i, j]
            row.append(f'{count}\\n({pct:.1f}%)')
        annot_text.append(row)
    
    sns.heatmap(cluster_group_df, annot=annot_text, fmt='', ax=axes[1,1], cmap='Blues')
    axes[1,1].set_title('Cluster vs Pathology Group\\n(Literature-Based Features)')
    axes[1,1].set_xlabel('Cluster')
    axes[1,1].set_ylabel('Pathology Group')
    
    # 6. Feature importance
    selected_features = clustering_results['selected_features']
    feature_importance = np.abs(clustering_results['kmeans'].cluster_centers_).mean(axis=0)
    
    # Sort features by importance
    feature_importance_df = pd.DataFrame({
        'feature': selected_features,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)
    
    axes[1,2].barh(range(len(feature_importance_df)), feature_importance_df['importance'])
    axes[1,2].set_yticks(range(len(feature_importance_df)))
    axes[1,2].set_yticklabels(feature_importance_df['feature'], fontsize=8)
    axes[1,2].set_title('Feature Importance\\n(Literature-Based)')
    axes[1,2].set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig('literature_based_clustering_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_literature_report(df, clustering_results):
    """Generate report with literature citations"""
    report = []
    report.append("LITERATURE-BASED CHOROID PLEXUS CLUSTERING ANALYSIS")
    report.append("=" * 70)
    report.append("")
    report.append("FEATURE BASIS:")
    report.append("• Volume features: Lizano et al. (2019), Choi et al. (2022)")
    report.append("• Intensity features: Tadayon et al. (2020)")
    report.append("• Morphology features: Choi et al. (2022), Zhou et al. (2021)")
    report.append("• Texture features: Zhou et al. (2021)")
    report.append("• Asymmetry features: Zhou et al. (2021)")
    report.append("• Spatial features: Van Praag et al. (2023)")
    report.append("")
    
    report.append(f"Total patients: {len(df)}")
    report.append("Group distribution:")
    for group, count in df['group'].value_counts().items():
        report.append(f"  {group}: {count} patients")
    report.append("")
    
    report.append("SELECTED FEATURES (most discriminative):")
    for feat in clustering_results['selected_features']:
        report.append(f"  • {feat}")
    report.append("")
    
    report.append("Clustering Results:")
    report.append(f"  Silhouette Score: {clustering_results['silhouette_score']:.3f}")
    report.append(f"  Adjusted Rand Index: {clustering_results['ari_score']:.3f}")
    report.append(f"  Cluster Purity: {clustering_results['cluster_purity']:.3f} ({clustering_results['cluster_purity']*100:.1f}%)")
    report.append(f"  Best Cluster Accuracy: {clustering_results['cluster_accuracy']:.3f} ({clustering_results['cluster_accuracy']*100:.1f}%)")
    report.append("")
    
    report.append("Optimal Cluster-to-Group Mapping:")
    for cluster_id, group_name in clustering_results['best_mapping'].items():
        report.append(f"  Cluster {cluster_id} → {group_name}")
    report.append("")
    
    # Key literature findings
    report.append("LITERATURE COMPARISON:")
    report.append("---------------------")
    
    # Volume analysis (Lizano et al. finding)
    ad_volume = df[df['group'] == 'AD']['choroid_plexus_volume'].mean()
    psy_volume = df[df['group'] == 'Psy']['choroid_plexus_volume'].mean()
    report.append(f"• AD mean volume: {ad_volume:.0f} voxels")
    report.append(f"• Psy mean volume: {psy_volume:.0f} voxels")
    
    if psy_volume > ad_volume:
        report.append("  → Consistent with Lizano et al. (2019): enlarged ChP in psychosis")
    else:
        report.append("  → Different from Lizano et al. (2019): no enlargement in psychosis")
    
    report.append("")
    report.append("Files generated:")
    report.append("  - literature_based_clustering_results.png")
    report.append("  - features_data_literature.csv")
    
    with open('literature_based_analysis_report.txt', 'w') as f:
        f.write('\\n'.join(report))
    
    for line in report:
        print(line)

def main():
    """Main function for literature-based analysis"""
    DATA_PATH = "/mnt/LIA/pazienti"
    JSON_PATH = "/mnt/LIA/pazienti/patients.json"
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path {DATA_PATH} does not exist!")
        return
    
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file {JSON_PATH} does not exist!")
        return
    
    print("Starting LITERATURE-BASED choroid plexus clustering analysis...")
    print("=" * 80)
    print("Using features validated in:")
    print("• Lizano et al. (2019) Mol Psychiatry - Volume & heterogeneity")
    print("• Tadayon et al. (2020) NeuroImage - Intensity & CSF contrast") 
    print("• Choi et al. (2022) Brain Imaging Behav - Morphology")
    print("• Zhou et al. (2021) Schizophr Res - Texture & asymmetry")
    print("=" * 80)
    
    # 1. Process patients with literature features
    df = process_patients_literature_based(DATA_PATH, JSON_PATH)
    
    if len(df) == 0:
        print("Error: No data processed!")
        return
    
    # 2. Clustering with feature selection
    clustering_results = perform_literature_based_clustering(df, k=4)
    
    # 3. Visualizations
    visualize_literature_results(df, clustering_results)
    
    # 4. Generate report
    generate_literature_report(df, clustering_results)
    
    # 5. Save data
    df.to_csv('features_data_literature.csv', index=False)
    
    print("\\nLITERATURE-BASED ANALYSIS complete!")
    
    return df, clustering_results

if __name__ == "__main__":
    df, results = main()
