"""
Unified K-means Clustering Analysis for Choroid Plexus Dataset
Combines raw image features AND segmentation mask features for comprehensive analysis.

This script performs clustering using both image characteristics and anatomical features
to provide a complete view of dataset complexity.

NEW VISUALIZATION FEATURES:
1. Traditional K-means cluster coloring (original approach)
2. True pathology group coloring with K-means cluster symbols
3. True pathology group coloring with K-means cluster color variations
4. Optional cluster decision boundaries (experimental)

Key Parameters for visualize_comparative_results():
- color_by_true_groups: Color points by true pathology groups instead of K-means clusters
- use_symbols_for_clusters: Use different symbols (circle, square, triangle, etc.) for K-means clusters
- draw_cluster_boundaries: Add decision boundary lines between clusters (can be visually messy)
- show_accuracy_borders: Green borders for correctly classified points

This allows switching between different visualization approaches to better understand
the relationship between true pathology groups and K-means cluster assignments.
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
from scipy.ndimage import uniform_filter

def plot_decision_boundaries(ax, X_pca, kmeans_model, h=0.02):
    """Plot the decision boundaries for k-means clusters in PCA space"""
    try:
        # Create a mesh to plot the decision boundary
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Create mesh points in PCA space
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # We need to inverse transform to original space, predict, then transform back
        # This is an approximation since we can't perfectly inverse PCA
        # Instead, we'll use distance-based assignment in PCA space
        
        # Get cluster centers in PCA space (approximate)
        cluster_centers_original = kmeans_model.cluster_centers_
        
        # For simplicity, we'll assign each mesh point to the nearest cluster center
        # by transforming the centers to PCA space
        from sklearn.metrics.pairwise import euclidean_distances
        
        # Calculate distances from each mesh point to each cluster in PCA space
        distances = []
        for center in cluster_centers_original:
            # This is a simplified approach - ideally we'd properly transform
            center_pca = center[:2] if len(center) >= 2 else np.array([0, 0])
            dist = euclidean_distances(mesh_points, center_pca.reshape(1, -1))
            distances.append(dist.flatten())
        
        # Assign to nearest cluster
        Z = np.argmin(distances, axis=0)
        Z = Z.reshape(xx.shape)
        
        # Plot the decision boundary
        ax.contour(xx, yy, Z, levels=range(len(cluster_centers_original)), 
                  alpha=0.3, colors='black', linestyles='dashed', linewidths=1)
        
    except Exception as e:
        print(f"Warning: Could not draw decision boundaries: {e}")

def calculate_cluster_accuracy(true_groups, cluster_labels):
    """Calculate cluster purity and accuracy metrics"""
    from scipy.optimize import linear_sum_assignment
    
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
        purity = max_group_count / total_in_cluster if total_in_cluster > 0 else 0
        
        cluster_purities.append(purity)
        cluster_sizes.append(total_in_cluster)
    
    # Overall purity
    total_samples = sum(cluster_sizes)
    overall_purity = sum(p * s for p, s in zip(cluster_purities, cluster_sizes)) / total_samples
    
    # Best accuracy using Hungarian algorithm
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
    return overall_purity, best_accuracy, best_mapping, confusion_matrix

def load_patient_groups(json_path):
    """Load patient group mappings from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['index_order_all']

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

def extract_raw_features_only(image_data):
    """Extract ONLY raw image features (NO segmentation info)"""
    features = {}
    
    # Normalize image first
    non_zero_mask = image_data > 0
    if np.sum(non_zero_mask) == 0:
        image_normalized = image_data
    else:
        p1, p99 = np.percentile(image_data[non_zero_mask], [1, 99])
        if p99 > p1:
            image_normalized = np.clip((image_data - p1) / (p99 - p1), 0, 1)
        else:
            image_normalized = image_data / np.max(image_data) if np.max(image_data) > 0 else image_data
    
    # Basic intensity statistics
    features['mean_intensity'] = np.mean(image_normalized)
    features['std_intensity'] = np.std(image_normalized)
    features['median_intensity'] = np.median(image_normalized)
    features['intensity_range'] = np.max(image_normalized) - np.min(image_normalized)
    
    # Intensity distribution features
    features['intensity_skewness'] = calculate_skewness(image_normalized.flatten())
    features['intensity_kurtosis'] = calculate_kurtosis(image_normalized.flatten())
    
    # Percentiles
    features['p25_intensity'] = np.percentile(image_normalized, 25)
    features['p75_intensity'] = np.percentile(image_normalized, 75)
    features['iqr_intensity'] = features['p75_intensity'] - features['p25_intensity']
    
    # Texture features (gradients)
    grad_x = ndimage.sobel(image_normalized, axis=0)
    grad_y = ndimage.sobel(image_normalized, axis=1)
    grad_z = ndimage.sobel(image_normalized, axis=2)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    features['gradient_mean'] = np.mean(gradient_magnitude)
    features['gradient_std'] = np.std(gradient_magnitude)
    
    # Laplacian (texture/edges)
    laplacian = ndimage.laplace(image_normalized.astype(float))
    features['laplacian_mean'] = np.mean(np.abs(laplacian))
    features['laplacian_std'] = np.std(laplacian)
    
    # Local variance (texture measure)
    kernel_size = 3
    local_mean = uniform_filter(image_normalized.astype(float), kernel_size)
    local_variance = uniform_filter(image_normalized**2, kernel_size) - local_mean**2
    features['local_variance_mean'] = np.mean(local_variance)
    features['local_variance_std'] = np.std(local_variance)
    
    # Entropy
    hist, _ = np.histogram(image_normalized, bins=256, density=True)
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist * np.log2(hist))
    
    return features

def extract_unified_features(image_data, mask_data):
    """Extract both raw image features AND segmentation mask features"""
    features = {}
    
    # ===== RAW IMAGE FEATURES =====
    # Normalize image first
    non_zero_mask = image_data > 0
    if np.sum(non_zero_mask) == 0:
        image_normalized = image_data
    else:
        p1, p99 = np.percentile(image_data[non_zero_mask], [1, 99])
        if p99 > p1:
            image_normalized = np.clip((image_data - p1) / (p99 - p1), 0, 1)
        else:
            image_normalized = image_data / np.max(image_data) if np.max(image_data) > 0 else image_data
    
    # Basic intensity statistics (on normalized data)
    features['mean_intensity'] = np.mean(image_normalized)
    features['std_intensity'] = np.std(image_normalized)
    features['median_intensity'] = np.median(image_normalized)
    features['intensity_range'] = np.max(image_normalized) - np.min(image_normalized)
    
    # Intensity distribution features
    features['intensity_skewness'] = calculate_skewness(image_normalized.flatten())
    features['intensity_kurtosis'] = calculate_kurtosis(image_normalized.flatten())
    
    # Percentiles
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
    
    # Laplacian (texture/edges)
    laplacian = ndimage.laplace(image_normalized.astype(float))
    features['laplacian_mean'] = np.mean(np.abs(laplacian))
    features['laplacian_std'] = np.std(laplacian)
    
    # Local variance (texture measure)
    kernel_size = 3
    local_mean = uniform_filter(image_normalized.astype(float), kernel_size)
    local_variance = uniform_filter(image_normalized**2, kernel_size) - local_mean**2
    features['local_variance_mean'] = np.mean(local_variance)
    features['local_variance_std'] = np.std(local_variance)
    
    # Entropy
    hist, _ = np.histogram(image_normalized, bins=256, density=True)
    hist = hist[hist > 0]
    features['entropy'] = -np.sum(hist * np.log2(hist))
    
    # ===== SEGMENTATION MASK FEATURES =====
    mask_binary = mask_data > 0
    features['volume'] = np.sum(mask_binary)
    
    if features['volume'] > 0:
        # Intensity within mask
        features['mask_mean_intensity'] = np.mean(image_normalized[mask_binary])
        features['mask_std_intensity'] = np.std(image_normalized[mask_binary])
        
        # Shape features
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
        
        # Compactness (sphericity measure)
        if features['surface_area'] > 0:
            features['compactness'] = (features['volume'] ** (2/3)) / features['surface_area']
        else:
            features['compactness'] = 0
            
    else:
        # Handle empty masks
        features.update({
            'mask_mean_intensity': 0,
            'mask_std_intensity': 0,
            'surface_area': 0,
            'surface_to_volume_ratio': 0,
            'num_components': 0,
            'centroid_x': 0, 'centroid_y': 0, 'centroid_z': 0,
            'compactness': 0
        })
    
    return features

def process_patients_raw_only(data_path, json_path):
    """Process all patients and extract RAW IMAGE features only"""
    print("Loading patient groups...")
    patient_groups = load_patient_groups(json_path)
    
    all_features = []
    
    print("Processing patients (RAW IMAGE FEATURES ONLY)...")
    for patient_id in tqdm(range(1, 105)):
        patient_folder = f"{patient_id:03d}"
        patient_path = os.path.join(data_path, patient_folder)
        
        if not os.path.exists(patient_path):
            continue
            
        # Only need the raw T1xFLAIR image
        image_file = f"{patient_folder}_T1xFLAIR.nii"
        image_path = os.path.join(patient_path, image_file)
        
        if not os.path.exists(image_path):
            continue
            
        try:
            # Load only the raw image
            image_nii = nib.load(image_path)
            image_data = image_nii.get_fdata()
            
            # Extract raw image features only
            features = extract_raw_features_only(image_data)
            features['patient_id'] = patient_id
            features['group'] = patient_groups.get(str(patient_id), 'unknown')
            
            all_features.append(features)
            
        except Exception as e:
            print(f"Error processing patient {patient_folder}: {e}")
            continue
    
    df = pd.DataFrame(all_features)
    print(f"Successfully processed {len(df)} patients (RAW FEATURES ONLY)")
    return df

def process_patients_unified(data_path, json_path):
    """Process all patients and extract unified features"""
    print("Loading patient groups...")
    patient_groups = load_patient_groups(json_path)
    
    all_features = []
    
    print("Processing patients (UNIFIED: Raw Image + Segmentation Features)...")
    for patient_id in tqdm(range(1, 105)):
        patient_folder = f"{patient_id:03d}"
        patient_path = os.path.join(data_path, patient_folder)
        
        if not os.path.exists(patient_path):
            continue
            
        # File paths
        image_file = f"{patient_folder}_T1xFLAIR.nii"
        mask_file = f"{patient_folder}_ChP_mask_T1xFLAIR_manual_seg.nii"
        
        image_path = os.path.join(patient_path, image_file)
        mask_path = os.path.join(patient_path, mask_file)
        
        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            continue
            
        try:
            # Load images
            image_nii = nib.load(image_path)
            mask_nii = nib.load(mask_path)
            
            image_data = image_nii.get_fdata()
            mask_data = mask_nii.get_fdata()
            
            # Extract unified features
            features = extract_unified_features(image_data, mask_data)
            features['patient_id'] = patient_id
            features['group'] = patient_groups.get(str(patient_id), 'unknown')
            
            all_features.append(features)
            
        except Exception as e:
            print(f"Error processing patient {patient_folder}: {e}")
            continue
    
    df = pd.DataFrame(all_features)
    print(f"Successfully processed {len(df)} patients (UNIFIED FEATURES)")
    return df

def perform_kmeans_analysis_raw(df, k=4):
    """Perform K-means clustering on raw image features"""
    print(f"Performing K-means clustering on RAW IMAGE FEATURES with k={k}...")
    
    # Prepare features for clustering
    feature_cols = [col for col in df.columns 
                   if col not in ['patient_id', 'group']]
    X = df[feature_cols].values
    
    print(f"Using {len(feature_cols)} raw image features:")
    for feat in feature_cols[:5]:
        print(f"  • {feat}")
    if len(feature_cols) > 5:
        print(f"  ... and {len(feature_cols)-5} more")
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
    
    # Calculate clustering metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    
    # Compare clusters with actual groups
    group_mapping = {'AD': 0, 'MCI': 1, 'Psy': 2, 'other': 3}
    true_labels = [group_mapping.get(group, 3) for group in df['group']]
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    
    # Calculate cluster purity and accuracy
    cluster_purity, cluster_accuracy, best_mapping, confusion_matrix = calculate_cluster_accuracy(df['group'], cluster_labels)
    
    print(f"RAW FEATURES - Silhouette Score: {silhouette_avg:.3f}")
    print(f"RAW FEATURES - Adjusted Rand Index: {ari_score:.3f}")
    print(f"RAW FEATURES - Cluster Purity: {cluster_purity:.3f} ({cluster_purity*100:.1f}%)")
    print(f"RAW FEATURES - Best Cluster Accuracy: {cluster_accuracy:.3f} ({cluster_accuracy*100:.1f}%)")
    print()
    
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
        'feature_cols': feature_cols,
        'confusion_matrix': confusion_matrix
    }

def perform_kmeans_analysis_unified(df, k=4):
    """Perform K-means clustering on unified features"""
    print(f"Performing K-means clustering on UNIFIED FEATURES with k={k}...")
    
    # Prepare features for clustering
    feature_cols = [col for col in df.columns 
                   if col not in ['patient_id', 'group']]
    X = df[feature_cols].values
    
    print(f"Using {len(feature_cols)} unified features:")
    raw_features = [f for f in feature_cols if 'mask' not in f and f not in ['volume', 'surface_area', 'surface_to_volume_ratio', 'num_components', 'centroid_x', 'centroid_y', 'centroid_z', 'compactness']]
    mask_features = [f for f in feature_cols if f not in raw_features]
    
    print(f"  Raw image features ({len(raw_features)}): {raw_features[:5]}..." if len(raw_features) > 5 else f"  Raw image features: {raw_features}")
    print(f"  Segmentation features ({len(mask_features)}): {mask_features}")
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
    
    # Calculate clustering metrics
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    
    # Compare clusters with actual groups
    group_mapping = {'AD': 0, 'MCI': 1, 'Psy': 2, 'other': 3}
    true_labels = [group_mapping.get(group, 3) for group in df['group']]
    ari_score = adjusted_rand_score(true_labels, cluster_labels)
    
    # Calculate cluster purity and accuracy
    cluster_purity, cluster_accuracy, best_mapping, confusion_matrix = calculate_cluster_accuracy(df['group'], cluster_labels)
    
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
        'feature_cols': feature_cols,
        'confusion_matrix': confusion_matrix
    }

def visualize_comparative_results(df_raw, df_unified, raw_results, unified_results, show_accuracy_borders=True, plot_first_four_only=True, color_by_true_groups=False, use_symbols_for_clusters=True, draw_cluster_boundaries=False, show_pie_chart=True, show_boxplot=True, show_clustering_plots=True, minimal_titles=None):
    """Create comprehensive comparison visualization
    
    Args:
        plot_first_four_only: If True, create 2x2 layout with only first 4 plots
                              If False, create 3x2 layout with all 6 plots (default)
        color_by_true_groups: If True, color points by true pathology groups and use markers for k-means clusters
                             If False, use traditional k-means cluster coloring (default)
        use_symbols_for_clusters: If True and color_by_true_groups=True, use different symbols for k-means clusters
                                 If False and color_by_true_groups=True, use same symbol but different colors
        draw_cluster_boundaries: If True, draw decision boundary lines between clusters (can be visually messy)
        show_accuracy_borders: If True, Green borders for correctly classified points
        minimal_titles: If True, use minimal titles. If None, auto-detect based on plot configuration

    This allows switching between different visualization approaches to better understand
    the relationship between true pathology groups and K-means cluster assignments.
    """
    # Determine layout based on what plots to show
    plots_to_show = []
    if show_pie_chart:
        plots_to_show.append('pie')
    if show_boxplot:
        plots_to_show.append('boxplot')
    
    if show_clustering_plots:
        plots_to_show.extend(['raw_pca', 'raw_confusion'])
        if not plot_first_four_only:
            plots_to_show.extend(['unified_pca', 'unified_confusion'])
    
    n_plots = len(plots_to_show)
    
    if n_plots == 0:
        print("Warning: No plots selected to show!")
        return
    
    # AUTO-ENABLE decision boundaries when showing only clustering plots
    minimal_clustering_only = show_clustering_plots and not show_pie_chart and not show_boxplot
    if minimal_clustering_only:
        draw_cluster_boundaries = True
        print("Auto-enabled decision boundaries for minimal clustering visualization")
        
        # Auto-enable minimal titles for clustering-only plots if not explicitly set
        if minimal_titles is None:
            minimal_titles = True
            print("Auto-enabled minimal titles for clustering-only visualization")
    
    # Set default minimal_titles if not specified
    if minimal_titles is None:
        minimal_titles = False
    
    # Determine figure layout
    if n_plots == 1:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]  # Make it a list for consistency
    elif n_plots == 2:  # Only PCA and confusion matrix OR only pie + boxplot
        if show_clustering_plots and not (show_pie_chart or show_boxplot):
            # Only clustering plots - adjust figure size and spacing
            fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        else:
            # Pie + boxplot or other combinations
            fig, axes = plt.subplots(1, 2, figsize=(12, 8))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes] if not isinstance(axes, list) else axes
    elif n_plots == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes = axes.flatten()
    elif n_plots == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
    elif n_plots == 5:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        # Hide the last subplot
        axes[5].set_visible(False)
    else:  # n_plots == 6
        fig, axes = plt.subplots(3, 2, figsize=(12, 18))
        axes = axes.flatten()
    
    # Color scheme
    cluster_colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12']  # Red, Green, Blue, Orange
    
    # Define colors for true groups and markers for k-means clusters (only if clustering plots are shown)
    if color_by_true_groups and show_clustering_plots:
        # Color palette for true pathology groups
        unique_groups = sorted(df_raw['group'].unique())
        group_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))[:len(unique_groups)]
        group_color_map = dict(zip(unique_groups, group_colors))
        
        if use_symbols_for_clusters:
            # Marker styles for k-means clusters
            cluster_markers = ['o', 's', '^', 'D', 'h', '*', 'v', '<', '>', 'p']
        else:
            # Use same marker but different cluster colors when not using symbols
            cluster_markers = ['o'] * 10  # All circles
            # Create cluster color map for alternative approach
            unique_clusters = sorted(df_raw['cluster'].unique())
            cluster_color_variations = plt.cm.Dark2(np.linspace(0, 1, len(unique_clusters)))
            cluster_color_map = dict(zip(unique_clusters, cluster_color_variations))
    
    plot_idx = 0
    subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    # Plot 1: Class Distribution (Pie Chart) - Optional
    if show_pie_chart:
        group_counts = df_unified['group'].value_counts()
        
        # Use same colors as scatter plot if color_by_true_groups is enabled and clustering plots are shown
        if color_by_true_groups and show_clustering_plots:
            # Ensure the pie chart colors match the group colors from scatter plots
            pie_colors = [group_color_map[group] for group in group_counts.index]
        else:
            # Use original colors
            pie_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        axes[plot_idx].pie(group_counts.values, labels=group_counts.index, autopct='%1.1f%%',
                          colors=pie_colors, textprops={'fontsize': 14})
        axes[plot_idx].set_title('Pathology Group Distribution', fontsize=16)
        axes[plot_idx].text(-0.1, 1.02, subplot_labels[plot_idx], transform=axes[plot_idx].transAxes, fontsize=16, fontweight='bold',
                           verticalalignment='bottom')
        plot_idx += 1
    
    # Plot 2: Mean Intensity and Volume by Group (dual y-axis boxplots) - Optional
    if show_boxplot:
        groups = ['AD', 'MCI', 'Psy', 'other']
        
        # Create dual y-axis plot for proper scaling
        ax1 = axes[plot_idx]
        ax2 = ax1.twinx()
        
        # Prepare data for boxplots
        intensity_data = []
        volume_data = []
        group_labels = []
        
        for g in groups:
            if g in df_unified['group'].values:
                intensity_data.append(df_unified[df_unified['group'] == g]['mean_intensity'].values)
                volume_data.append(df_unified[df_unified['group'] == g]['volume'].values)
                group_labels.append(g)
        
        # Create side-by-side boxplots with tighter group spacing
        base_positions = np.arange(len(group_labels)) * 0.7  # Compress group spacing
        intensity_positions = base_positions - 0.15  # Shift intensity boxes to the left (closer)
        volume_positions = base_positions + 0.15     # Shift volume boxes to the right (closer)
        box_width = 0.25  # Much narrower boxes for tighter display
        
        # Intensity boxplots (left y-axis) - positioned to the left of center
        bp1 = ax1.boxplot(intensity_data, positions=intensity_positions, widths=box_width, 
                          patch_artist=True, showfliers=True)
        
        # Volume boxplots (right y-axis) - positioned to the right of center
        bp2 = ax2.boxplot(volume_data, positions=volume_positions, widths=box_width,
                          patch_artist=True, showfliers=True)
        
        # Color and style the boxes
        for patch in bp1['boxes']:
            patch.set_facecolor('steelblue')
            patch.set_alpha(0.7)
            patch.set_edgecolor('navy')
        
        for patch in bp2['boxes']:
            patch.set_facecolor('coral')
            patch.set_alpha(0.7)
            patch.set_edgecolor('darkred')
        
        # Set labels and titles
        ax1.set_xticks(base_positions)
        ax1.set_xticklabels(group_labels)
        ax1.set_ylabel('Mean Intensity (normalized)', color='steelblue', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Volume (voxels)', color='coral', fontweight='bold', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=12)
        ax2.tick_params(axis='y', labelcolor='coral', labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)
        
        ax1.set_title('Mean Intensity & Volume\nDistribution by Group', fontsize=16)
        
        # Squeeze the boxplot width by setting tighter axis limits for compressed spacing
        ax1.set_xlim(-0.3, (len(group_labels) - 1) * 0.7 + 0.3)
        
        # Add legend with smaller size
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor='steelblue', alpha=0.7, label='Mean Intensity (normalized)'),
                          plt.Rectangle((0,0),1,1, facecolor='coral', alpha=0.7, label='Volume (voxels)')]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=13)
        
        # Add subplot label
        axes[plot_idx].text(-0.1, 1.02, subplot_labels[plot_idx], transform=axes[plot_idx].transAxes, fontsize=16, fontweight='bold',
                           verticalalignment='bottom')
        
        plot_idx += 1

    # Only add clustering plots if requested
    if show_clustering_plots:
        # Plot 3: Raw Features K-means PCA
        X_scaled_raw = raw_results['X_scaled']
        pca_raw = PCA(n_components=2)
        X_pca_raw = pca_raw.fit_transform(X_scaled_raw)
        
        best_mapping_raw = raw_results['best_mapping']
        unique_clusters_raw = sorted(df_raw['cluster'].unique())
        
        # Plot PCA results with different styles based on options
        if color_by_true_groups:
            if use_symbols_for_clusters:
                # Method 1: Color by true groups, use different symbols for k-means clusters
                for group in unique_groups:
                    for i, cluster in enumerate(unique_clusters_raw):
                        group_cluster_mask = (df_raw['group'] == group) & (df_raw['cluster'] == cluster)
                        if group_cluster_mask.any():
                            indices = df_raw[group_cluster_mask].index
                            pca_indices = [idx for idx in range(len(df_raw)) if df_raw.index[idx] in indices]
                            
                            if show_accuracy_borders:
                                optimal_group = best_mapping_raw.get(cluster, None)
                                edgecolor = 'green' if group == optimal_group else 'black'
                                linewidth = 2 if group == optimal_group else 0.5
                            else:
                                edgecolor = 'black'
                                linewidth = 0.5
                            
                            axes[plot_idx].scatter(X_pca_raw[pca_indices, 0], X_pca_raw[pca_indices, 1], 
                                            c=[group_color_map[group]], marker=cluster_markers[i % len(cluster_markers)], 
                                            s=80, alpha=0.8, edgecolors=edgecolor, linewidths=linewidth)
                
                # Create legend for groups and markers
                group_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                                  markerfacecolor=group_color_map[group], markersize=8, 
                                                  label=f'{group}') for group in unique_groups]
                cluster_legend_elements = [plt.Line2D([0], [0], marker=cluster_markers[i % len(cluster_markers)], 
                                                    color='w', markerfacecolor='gray', markersize=8, 
                                                    label=f'Cluster {cluster}') for i, cluster in enumerate(unique_clusters_raw)]
                
                legend1 = axes[plot_idx].legend(handles=group_legend_elements, title='True Groups', 
                                         loc='upper right', fontsize=10)
                axes[plot_idx].add_artist(legend1)
                axes[plot_idx].legend(handles=cluster_legend_elements, title='K-means Clusters', 
                               loc='lower right', fontsize=10)
            else:
                # Method 2: Color by true groups, use color variations for k-means clusters
                for group in unique_groups:
                    group_mask = df_raw['group'] == group
                    group_data = df_raw[group_mask]
                    
                    for i, cluster in enumerate(unique_clusters_raw):
                        cluster_mask = group_data['cluster'] == cluster
                        if cluster_mask.any():
                            cluster_indices = group_data[cluster_mask].index
                            pca_indices = [idx for idx in range(len(df_raw)) if df_raw.index[idx] in cluster_indices]
                            
                            # Blend group color with cluster color for distinction
                            base_color = group_color_map[group]
                            cluster_variation = cluster_color_map[cluster]
                            # Mix colors: 70% group color + 30% cluster variation
                            mixed_color = 0.7 * np.array(base_color[:3]) + 0.3 * np.array(cluster_variation[:3])
                            
                            if show_accuracy_borders:
                                optimal_group = best_mapping_raw.get(cluster, None)
                                edgecolor = 'green' if group == optimal_group else 'black'
                                linewidth = 2 if group == optimal_group else 0.5
                            else:
                                edgecolor = 'black'
                                linewidth = 0.5
                            
                            axes[plot_idx].scatter(X_pca_raw[pca_indices, 0], X_pca_raw[pca_indices, 1], 
                                            c=[mixed_color], marker='o', s=80, alpha=0.8, 
                                            edgecolors=edgecolor, linewidths=linewidth)
                
                # Create simplified legend for groups only
                group_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                                  markerfacecolor=group_color_map[group], markersize=8, 
                                                  label=f'{group}') for group in unique_groups]
                axes[plot_idx].legend(handles=group_legend_elements, title='True Groups', 
                               loc='upper right', fontsize=12)
        else:
            # Original cluster-based coloring
            for i, cluster in enumerate(unique_clusters_raw):
                mask = df_raw['cluster'] == cluster
                
                if show_accuracy_borders:
                    optimal_group = best_mapping_raw.get(cluster, None)
                    cluster_data = df_raw[mask]
                    correct_mask = cluster_data['group'] == optimal_group
                    incorrect_mask = ~correct_mask
                    
                    if correct_mask.any():
                        correct_indices = cluster_data[correct_mask].index
                        correct_pca_indices = [idx for idx in range(len(df_raw)) if df_raw.index[idx] in correct_indices]
                        axes[plot_idx].scatter(X_pca_raw[correct_pca_indices, 0], X_pca_raw[correct_pca_indices, 1], 
                                        c=cluster_colors[i], s=60, alpha=0.8,
                                        edgecolors='green', linewidths=2)
                    
                    if incorrect_mask.any():
                        incorrect_indices = cluster_data[incorrect_mask].index
                        incorrect_pca_indices = [idx for idx in range(len(df_raw)) if df_raw.index[idx] in incorrect_indices]
                        axes[plot_idx].scatter(X_pca_raw[incorrect_pca_indices, 0], X_pca_raw[incorrect_pca_indices, 1], 
                                        c=cluster_colors[i], s=60, alpha=0.8)
                    
                    # Add legend entry for this cluster
                    axes[plot_idx].scatter([], [], c=cluster_colors[i], label=f'Cluster {cluster}', 
                                    alpha=0.8, s=60)
                
                else:
                    axes[plot_idx].scatter(X_pca_raw[mask, 0], X_pca_raw[mask, 1], 
                                   c=cluster_colors[i], label=f'Cluster {cluster}', 
                                   alpha=0.8, s=60)
    
        # Draw cluster decision boundaries if requested
        if draw_cluster_boundaries:
            # Fit a new KMeans model on the PCA data for boundary visualization
            kmeans_pca_raw = KMeans(n_clusters=len(unique_clusters_raw), random_state=42)
            kmeans_pca_raw.fit(X_pca_raw)
            plot_decision_boundaries(axes[plot_idx], X_pca_raw, kmeans_pca_raw)
        
        # Set title based on minimal_titles setting
        if minimal_titles:
            title_raw = 'Raw Image Features'
        else:
            title_raw = 'K-means Clusters (PCA)\nRAW IMAGE FEATURES ONLY'
            if color_by_true_groups:
                title_raw = 'True Groups & K-means Clusters (PCA)\nRAW IMAGE FEATURES ONLY'
                if use_symbols_for_clusters:
                    title_raw += '\nColors: True groups, Symbols: K-means clusters'
                else:
                    title_raw += '\nColors: True groups with K-means variations'
            if show_accuracy_borders:
                title_raw += '\nGreen border: Correctly classified'
            if draw_cluster_boundaries:
                title_raw += '\nDashed lines: Cluster boundaries'
        
        axes[plot_idx].set_title(title_raw, fontsize=14)
        axes[plot_idx].set_xlabel(f'PC1 ({pca_raw.explained_variance_ratio_[0]:.1%} variance)', fontsize=13)
        axes[plot_idx].set_ylabel(f'PC2 ({pca_raw.explained_variance_ratio_[1]:.1%} variance)', fontsize=13)
        
        # Only add default legend if not using group coloring (group coloring adds custom legends above)
        if not color_by_true_groups:
            axes[plot_idx].legend(title='Clusters', loc='upper right', fontsize=12)
        
        axes[plot_idx].tick_params(labelsize=11)
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Fix subplot label positioning for horizontal layouts
        if n_plots == 2 and minimal_clustering_only:
            # For 2-plot horizontal layout, use consistent positioning
            axes[plot_idx].text(-0.15, 1.05, subplot_labels[plot_idx], transform=axes[plot_idx].transAxes, 
                               fontsize=16, fontweight='bold', verticalalignment='bottom')
        else:
            axes[plot_idx].text(-0.1, 1.02, subplot_labels[plot_idx], transform=axes[plot_idx].transAxes, 
                               fontsize=16, fontweight='bold', verticalalignment='bottom')
        
        plot_idx += 1
        
        # Plot 4: Raw Features Confusion Matrix
        confusion_matrix_raw = raw_results['confusion_matrix']
        
        total_samples_raw = confusion_matrix_raw.sum().sum()
        annot_matrix_raw = np.zeros_like(confusion_matrix_raw.values, dtype=object)
        
        for i in range(confusion_matrix_raw.shape[0]):
            for j in range(confusion_matrix_raw.shape[1]):
                count = confusion_matrix_raw.iloc[i, j]
                pct = (count / total_samples_raw) * 100
                annot_matrix_raw[i, j] = f'{count}\n({pct:.1f}%)'
        
        sns.heatmap(confusion_matrix_raw, annot=annot_matrix_raw, fmt='', ax=axes[plot_idx], 
                    cmap='Blues', cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
        
        # Set title based on minimal_titles setting
        if minimal_titles:
            axes[plot_idx].set_title('Confusion Matrix', fontsize=14)
        else:
            axes[plot_idx].set_title('Confusion Matrix\nRAW FEATURES', fontsize=15)
            
        axes[plot_idx].set_xlabel('True Pathology Group', fontsize=13)
        axes[plot_idx].set_ylabel('Predicted Cluster', fontsize=13)
        axes[plot_idx].tick_params(labelsize=12)
        
        # Fix subplot label positioning for horizontal layouts
        if n_plots == 2 and minimal_clustering_only:
            # For 2-plot horizontal layout, use consistent positioning
            axes[plot_idx].text(-0.15, 1.05, subplot_labels[plot_idx], transform=axes[plot_idx].transAxes, 
                               fontsize=16, fontweight='bold', verticalalignment='bottom')
        else:
            axes[plot_idx].text(-0.1, 1.02, subplot_labels[plot_idx], transform=axes[plot_idx].transAxes, 
                               fontsize=16, fontweight='bold', verticalalignment='bottom')
        
        # Adjust confusion matrix layout
        axes[plot_idx].set_aspect('equal')
        
        plot_idx += 1
    
    # Skip plots 5 and 6 if only plotting first four
    if plot_first_four_only or not show_clustering_plots:
        # Adjust subplot spacing based on layout
        if n_plots == 1:
            plt.subplots_adjust(left=0.15, right=0.85, top=0.90, bottom=0.15)
        elif n_plots == 2:
            if minimal_clustering_only:
                # Special spacing for clustering-only plots with better alignment
                if minimal_titles:
                    plt.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.15, wspace=0.25)
                else:
                    plt.subplots_adjust(left=0.05, right=0.98, top=0.80, bottom=0.12, wspace=0.20)
            else:
                plt.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.10, wspace=0.20)
        elif n_plots == 3:
            plt.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.10, wspace=0.15)
        else:  # n_plots == 4
            plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05, wspace=0.15, hspace=0.25)
        
        # Force boxplot to be narrower AFTER subplot adjustments (only if boxplot exists)
        if show_boxplot:
            # Find the boxplot index
            boxplot_idx = 1 if show_pie_chart else 0  # Find correct boxplot index
            pos1 = axes[boxplot_idx].get_position()
            axes[boxplot_idx].set_position([pos1.x0 + 0.01, pos1.y0, pos1.width * 0.88, pos1.height])
            # ax2 is the twin axis created earlier in the boxplot section
            if 'ax2' in locals():
                pos2 = ax2.get_position()  
                ax2.set_position([pos2.x0 + 0.01, pos2.y0, pos2.width * 0.88, pos2.height])
        
        # Generate appropriate filename
        suffix = ""
        if not show_pie_chart and not show_boxplot:
            suffix = "_minimal"
            if minimal_titles:
                suffix += "_clean"
        elif not show_pie_chart:
            suffix = "_no_pie"
        elif not show_boxplot:
            suffix = "_no_boxplot"
        elif not show_clustering_plots:
            suffix = "_descriptive_only"
        
        if plot_first_four_only:
            filename = f'comparative_kmeans_analysis_4plots{suffix}.png'
        else:
            filename = f'comparative_kmeans_analysis{suffix}.png'
            
        if color_by_true_groups and show_clustering_plots:
            if use_symbols_for_clusters and draw_cluster_boundaries:
                filename = filename.replace('.png', '_true_groups_symbols_boundaries.png')
            elif use_symbols_for_clusters:
                filename = filename.replace('.png', '_true_groups_symbols.png')
            elif draw_cluster_boundaries:
                filename = filename.replace('.png', '_true_groups_boundaries.png')
            else:
                filename = filename.replace('.png', '_true_groups_colors.png')
        plt.savefig(filename, dpi=300)
        plt.show()
        return
    
    # Continue with unified plots (plots 5 and 6) if not first_four_only and show_clustering_plots
    # Plot 5: Unified Features K-means PCA
    X_scaled_unified = unified_results['X_scaled']
    pca_unified = PCA(n_components=2)
    X_pca_unified = pca_unified.fit_transform(X_scaled_unified)
    
    best_mapping_unified = unified_results['best_mapping']
    unique_clusters_unified = sorted(df_unified['cluster'].unique())
    
    if color_by_true_groups:
        if use_symbols_for_clusters:
            # Method 1: Color by true groups, use different symbols for k-means clusters
            for group in unique_groups:
                for i, cluster in enumerate(unique_clusters_unified):
                    group_cluster_mask = (df_unified['group'] == group) & (df_unified['cluster'] == cluster)
                    if group_cluster_mask.any():
                        indices = df_unified[group_cluster_mask].index
                        pca_indices = [idx for idx in range(len(df_unified)) if df_unified.index[idx] in indices]
                        
                        if show_accuracy_borders:
                            optimal_group = best_mapping_unified.get(cluster, None)
                            edgecolor = 'green' if group == optimal_group else 'black'
                            linewidth = 2 if group == optimal_group else 0.5
                        else:
                            edgecolor = 'black'
                            linewidth = 0.5
                        
                        axes[plot_idx].scatter(X_pca_unified[pca_indices, 0], X_pca_unified[pca_indices, 1], 
                                        c=[group_color_map[group]], marker=cluster_markers[i % len(cluster_markers)], 
                                        s=80, alpha=0.8, edgecolors=edgecolor, linewidths=linewidth)
            
            # Create legend for groups and markers
            group_legend_elements_unified = [plt.Line2D([0], [0], marker='o', color='w', 
                                                      markerfacecolor=group_color_map[group], markersize=8, 
                                                      label=f'{group}') for group in unique_groups]
            cluster_legend_elements_unified = [plt.Line2D([0], [0], marker=cluster_markers[i % len(cluster_markers)], 
                                                        color='w', markerfacecolor='gray', markersize=8, 
                                                        label=f'Cluster {cluster}') for i, cluster in enumerate(unique_clusters_unified)]
            
            legend1_unified = axes[plot_idx].legend(handles=group_legend_elements_unified, title='True Groups', 
                                                 loc='upper right', fontsize=10)
            axes[plot_idx].add_artist(legend1_unified)
            axes[plot_idx].legend(handles=cluster_legend_elements_unified, title='K-means Clusters', 
                               loc='lower right', fontsize=10)
        else:
            # Method 2: Color by true groups, use color variations for k-means clusters
            for group in unique_groups:
                group_mask = df_unified['group'] == group
                group_data = df_unified[group_mask]
                
                for i, cluster in enumerate(unique_clusters_unified):
                    cluster_mask = group_data['cluster'] == cluster
                    if cluster_mask.any():
                        cluster_indices = group_data[cluster_mask].index
                        pca_indices = [idx for idx in range(len(df_unified)) if df_unified.index[idx] in cluster_indices]
                        
                        # Blend group color with cluster color for distinction
                        base_color = group_color_map[group]
                        cluster_variation = cluster_color_map[cluster]
                        # Mix colors: 70% group color + 30% cluster variation
                        mixed_color = 0.7 * np.array(base_color[:3]) + 0.3 * np.array(cluster_variation[:3])
                        
                        if show_accuracy_borders:
                            optimal_group = best_mapping_unified.get(cluster, None)
                            edgecolor = 'green' if group == optimal_group else 'black'
                            linewidth = 2 if group == optimal_group else 0.5
                        else:
                            edgecolor = 'black'
                            linewidth = 0.5
                        
                        axes[plot_idx].scatter(X_pca_unified[pca_indices, 0], X_pca_unified[pca_indices, 1], 
                                        c=[mixed_color], marker='o', s=80, alpha=0.8, 
                                        edgecolors=edgecolor, linewidths=linewidth)
            
            # Create simplified legend for groups only
            group_legend_elements_unified = [plt.Line2D([0], [0], marker='o', color='w', 
                                                      markerfacecolor=group_color_map[group], markersize=8, 
                                                      label=f'{group}') for group in unique_groups]
            axes[plot_idx].legend(handles=group_legend_elements_unified, title='True Groups', 
                           loc='upper right', fontsize=12)
    else:
        # Original cluster-based coloring
        for i, cluster in enumerate(unique_clusters_unified):
            mask = df_unified['cluster'] == cluster
            
            if show_accuracy_borders:
                optimal_group = best_mapping_unified.get(cluster, None)
                cluster_data = df_unified[mask]
                correct_mask = cluster_data['group'] == optimal_group
                incorrect_mask = ~correct_mask
                
                if correct_mask.any():
                    correct_indices = cluster_data[correct_mask].index
                    correct_pca_indices = [idx for idx in range(len(df_unified)) if df_unified.index[idx] in correct_indices]
                    axes[plot_idx].scatter(X_pca_unified[correct_pca_indices, 0], X_pca_unified[correct_pca_indices, 1], 
                                    c=cluster_colors[i], s=60, alpha=0.8,
                                    edgecolors='green', linewidths=2)
                
                if incorrect_mask.any():
                    incorrect_indices = cluster_data[incorrect_mask].index
                    incorrect_pca_indices = [idx for idx in range(len(df_unified)) if df_unified.index[idx] in incorrect_indices]
                    axes[plot_idx].scatter(X_pca_unified[incorrect_pca_indices, 0], X_pca_unified[incorrect_pca_indices, 1], 
                                    c=cluster_colors[i], s=60, alpha=0.8)
                
                # Add legend entry for this cluster
                axes[plot_idx].scatter([], [], c=cluster_colors[i], label=f'Cluster {cluster}', 
                                alpha=0.8, s=60)
            else:
                axes[plot_idx].scatter(X_pca_unified[mask, 0], X_pca_unified[mask, 1], 
                               c=cluster_colors[i], label=f'Cluster {cluster}', 
                               alpha=0.8, s=60)

    # Draw cluster decision boundaries if requested
    if draw_cluster_boundaries:
        # Fit a new KMeans model on the PCA data for boundary visualization
        kmeans_pca_unified = KMeans(n_clusters=len(unique_clusters_unified), random_state=42)
        kmeans_pca_unified.fit(X_pca_unified)
        plot_decision_boundaries(axes[plot_idx], X_pca_unified, kmeans_pca_unified)
    
    title_unified = 'K-means Clusters (PCA)\nRAW + SEGMENTATION FEATURES'
    if color_by_true_groups:
        title_unified = 'True Groups & K-means Clusters (PCA)\nRAW + SEGMENTATION FEATURES'
        if use_symbols_for_clusters:
            title_unified += '\nColors: True groups, Symbols: K-means clusters'
        else:
            title_unified += '\nColors: True groups with K-means variations'
    if show_accuracy_borders:
        title_unified += '\nGreen border: Correctly classified'
    if draw_cluster_boundaries:
        title_unified += '\nDashed lines: Cluster boundaries'
    
    axes[plot_idx].set_title(title_unified, fontsize=15)
    axes[plot_idx].set_xlabel(f'PC1 ({pca_unified.explained_variance_ratio_[0]:.1%} variance)', fontsize=13)
    axes[plot_idx].set_ylabel(f'PC2 ({pca_unified.explained_variance_ratio_[1]:.1%} variance)', fontsize=13)
    
    # Only add default legend if not using group coloring (group coloring adds custom legends above)
    if not color_by_true_groups:
        axes[plot_idx].legend(title='Clusters', loc='upper right', fontsize=12)
    
    axes[plot_idx].tick_params(labelsize=11)
    axes[plot_idx].grid(True, alpha=0.3)
    axes[plot_idx].text(-0.1, 1.02, subplot_labels[plot_idx], transform=axes[plot_idx].transAxes, fontsize=16, fontweight='bold',
                   verticalalignment='bottom')
    
    # Plot 6: Unified Features Confusion Matrix
    confusion_matrix_unified = unified_results['confusion_matrix']
    
    total_samples_unified = confusion_matrix_unified.sum().sum()
    annot_matrix_unified = np.zeros_like(confusion_matrix_unified.values, dtype=object)
    
    for i in range(confusion_matrix_unified.shape[0]):
        for j in range(confusion_matrix_unified.shape[1]):
            count = confusion_matrix_unified.iloc[i, j]
            pct = (count / total_samples_unified) * 100
            annot_matrix_unified[i, j] = f'{count}\n({pct:.1f}%)'
    
    sns.heatmap(confusion_matrix_unified, annot=annot_matrix_unified, fmt='', ax=axes[plot_idx], 
                cmap='Blues', cbar_kws={'label': 'Count'}, annot_kws={'size': 12})
    axes[plot_idx].set_title('Confusion Matrix\nRAW + SEGMENTATION FEATURES', fontsize=14)
    axes[plot_idx].set_xlabel('True Pathology Group', fontsize=12)
    axes[plot_idx].set_ylabel('Predicted Cluster', fontsize=12)
    axes[plot_idx].tick_params(labelsize=11)
    axes[plot_idx].text(-0.1, 1.02, subplot_labels[plot_idx], transform=axes[plot_idx].transAxes, fontsize=16, fontweight='bold',
                   verticalalignment='bottom')
    
    # Adjust confusion matrix layout
    axes[plot_idx].set_aspect('equal')
    
    # Adjust subplot spacing with better column separation and more row spacing
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05, wspace=0.15, hspace=0.3)
    
    # Force boxplot to be narrower AFTER subplot adjustments (less aggressive)
    # Fix: Use 1D indexing since axes is flattened
    if show_boxplot:
        boxplot_idx = 1 if show_pie_chart else 0  # Find correct boxplot index
        pos1 = axes[boxplot_idx].get_position()
        axes[boxplot_idx].set_position([pos1.x0 + 0.01, pos1.y0, pos1.width * 0.88, pos1.height])
        # ax2 is the twin axis created earlier in the boxplot section
        if 'ax2' in locals():
            pos2 = ax2.get_position()  
            ax2.set_position([pos2.x0 + 0.01, pos2.y0, pos2.width * 0.88, pos2.height])
    
    # Generate appropriate filename for full plots
    suffix = ""
    if not show_pie_chart and not show_boxplot:
        suffix = "_minimal"
    elif not show_pie_chart:
        suffix = "_no_pie"
    elif not show_boxplot:
        suffix = "_no_boxplot"
    
    filename = f'comparative_kmeans_analysis{suffix}.png'
    if color_by_true_groups:
        if use_symbols_for_clusters and draw_cluster_boundaries:
            filename = f'comparative_kmeans_analysis{suffix}_true_groups_symbols_boundaries.png'
        elif use_symbols_for_clusters:
            filename = f'comparative_kmeans_analysis{suffix}_true_groups_symbols.png'
        elif draw_cluster_boundaries:
            filename = f'comparative_kmeans_analysis{suffix}_true_groups_boundaries.png'
        else:
            filename = f'comparative_kmeans_analysis{suffix}_true_groups_colors.png'
    plt.savefig(filename, dpi=300)
    plt.show()

def generate_unified_report(df, clustering_results):
    """Generate comprehensive unified analysis report"""
    report = []
    report.append("UNIFIED K-MEANS CLUSTERING ANALYSIS")
    report.append("=" * 60)
    report.append("")
    report.append("ANALYSIS TYPE: Raw T1xFLAIR image + Choroid Plexus segmentation features")
    report.append("COMBINES: Image characteristics AND anatomical measurements")
    report.append("")
    
    # Dataset overview
    report.append(f"Total patients: {len(df)}")
    report.append("Group distribution:")
    for group, count in df['group'].value_counts().items():
        report.append(f"  {group}: {count} patients")
    report.append("")
    
    # Feature summary
    feature_count = len(clustering_results['feature_cols'])
    raw_features = [f for f in clustering_results['feature_cols'] 
                   if 'mask' not in f and f not in ['volume', 'surface_area', 'surface_to_volume_ratio', 
                                                   'num_components', 'centroid_x', 'centroid_y', 'centroid_z', 'compactness']]
    mask_features = [f for f in clustering_results['feature_cols'] if f not in raw_features]
    
    report.append(f"Total features used: {feature_count}")
    report.append(f"  Raw image features: {len(raw_features)}")
    report.append("    • Intensity statistics, texture measures, entropy")
    report.append(f"  Segmentation features: {len(mask_features)}")
    report.append("    • Volume, surface area, shape, connectivity")
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
        complexity = "HIGH - Strong separation between pathology groups"
    elif clustering_results['silhouette_score'] > 0.25:
        complexity = "MODERATE - Some distinct patterns exist"
    else:
        complexity = "LOW - Limited separability between groups"
    
    report.append(f"Dataset Complexity: {complexity}")
    report.append("")
    
    # Group statistics
    report.append("Group Statistics (mean ± std):")
    for group in ['AD', 'MCI', 'Psy', 'other']:
        group_data = df[df['group'] == group]
        if len(group_data) > 0:
            vol_mean = group_data['volume'].mean()
            vol_std = group_data['volume'].std()
            intensity_mean = group_data['mean_intensity'].mean()
            entropy_mean = group_data['entropy'].mean()
            report.append(f"  {group}: Vol={vol_mean:.0f}±{vol_std:.0f}, Int={intensity_mean:.3f}, Ent={entropy_mean:.2f}")
    
    report.append("")
    report.append("FILES GENERATED:")
    report.append("  - unified_kmeans_analysis.png (PCA clusters + confusion matrix)")
    report.append("  - unified_features_data.csv (all extracted features)")
    report.append("  - unified_analysis_report.txt (this report)")
    
    # Save report
    with open('unified_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Print report
    for line in report:
        print(line)

def generate_all_visualizations(df_raw, df_unified, raw_results, unified_results):
    """Generate all possible visualization combinations"""
    print("\n" + "="*70)
    print("GENERATING ALL VISUALIZATION COMBINATIONS")
    print("="*70)
    
    combinations = [
        # Basic combinations
        {
            "name": "1. Full visualization (all plots)",
            "params": {"show_pie_chart": True, "show_boxplot": True, "show_clustering_plots": True, "color_by_true_groups": False}
        },
        {
            "name": "2. Minimal visualization (PCA + Confusion Matrix only)",
            "params": {"show_pie_chart": False, "show_boxplot": False, "show_clustering_plots": True, "color_by_true_groups": False}
        },
        {
            "name": "2b. Minimal visualization - Clean titles",
            "params": {"show_pie_chart": False, "show_boxplot": False, "show_clustering_plots": True, "color_by_true_groups": False, "minimal_titles": True}
        },
        {
            "name": "3. Descriptive only (Pie Chart + Boxplot only)",
            "params": {"show_pie_chart": True, "show_boxplot": True, "show_clustering_plots": False, "color_by_true_groups": False}
        },
        {
            "name": "4. No pie chart (Boxplot + Clustering)",
            "params": {"show_pie_chart": False, "show_boxplot": True, "show_clustering_plots": True, "color_by_true_groups": False}
        },
        {
            "name": "5. No boxplot (Pie Chart + Clustering)",
            "params": {"show_pie_chart": True, "show_boxplot": False, "show_clustering_plots": True, "color_by_true_groups": False}
        },
        
        # True group coloring variations
        {
            "name": "6. True groups with symbols",
            "params": {"show_pie_chart": True, "show_boxplot": True, "show_clustering_plots": True, "color_by_true_groups": True, "use_symbols_for_clusters": True}
        },
        {
            "name": "7. True groups with color variations",
            "params": {"show_pie_chart": True, "show_boxplot": True, "show_clustering_plots": True, "color_by_true_groups": True, "use_symbols_for_clusters": False}
        },
        {
            "name": "8. True groups with decision boundaries",
            "params": {"show_pie_chart": True, "show_boxplot": True, "show_clustering_plots": True, "color_by_true_groups": True, "use_symbols_for_clusters": True, "draw_cluster_boundaries": True}
        },
        
        # Minimal with true groups
        {
            "name": "9. Minimal with true groups + symbols",
            "params": {"show_pie_chart": False, "show_boxplot": False, "show_clustering_plots": True, "color_by_true_groups": True, "use_symbols_for_clusters": True}
        },
        {
            "name": "9b. Minimal with true groups + symbols - Clean",
            "params": {"show_pie_chart": False, "show_boxplot": False, "show_clustering_plots": True, "color_by_true_groups": True, "use_symbols_for_clusters": True, "minimal_titles": True}
        },
        {
            "name": "10. Pie chart only",
            "params": {"show_pie_chart": True, "show_boxplot": False, "show_clustering_plots": False, "color_by_true_groups": False}
        },
        {
            "name": "11. Boxplot only",
            "params": {"show_pie_chart": False, "show_boxplot": True, "show_clustering_plots": False, "color_by_true_groups": False}
        }
    ]
    
    for combo in combinations:
        print(f"\nGenerating: {combo['name']}")
        try:
            visualize_comparative_results(
                df_raw, df_unified, raw_results, unified_results,
                show_accuracy_borders=True,
                plot_first_four_only=True,
                **combo['params']
            )
        except Exception as e:
            print(f"Error generating {combo['name']}: {e}")
    
    print(f"\n{'='*70}")
    print("ALL VISUALIZATIONS GENERATED!")
    print("="*70)
    print("Generated files:")
    print("  • comparative_kmeans_analysis_4plots.png - Full with K-means coloring")
    print("  • comparative_kmeans_analysis_4plots_minimal.png - PCA + Confusion only")
    print("  • comparative_kmeans_analysis_4plots_descriptive_only.png - Pie + Boxplot only")
    print("  • comparative_kmeans_analysis_4plots_no_pie.png - Boxplot + Clustering")
    print("  • comparative_kmeans_analysis_4plots_no_boxplot.png - Pie + Clustering")
    print("  • comparative_kmeans_analysis_4plots_true_groups_symbols.png - True groups + symbols")
    print("  • comparative_kmeans_analysis_4plots_true_groups_colors.png - True groups + color variations")
    print("  • comparative_kmeans_analysis_4plots_true_groups_symbols_boundaries.png - With boundaries")
    print("  • And more single-plot variations...")

def main(skip_processing=False, generate_all=False):
    """Main function for comparative analysis
    
    Args:
        skip_processing: If True, load existing CSV files instead of processing images
        generate_all: If True, generate all possible visualization combinations
    """
    # Configuration
    DATA_PATH = "/mnt/LIA/pazienti"
    JSON_PATH = "/mnt/LIA/pazienti/patients.json"
    
    # Check if we should skip processing and load existing files
    if skip_processing:
        print("SKIP MODE: Loading existing CSV files...")
        print("=" * 70)
        
        # Check if files exist
        if os.path.exists('raw_features_data.csv') and os.path.exists('unified_features_data.csv'):
            print("Loading existing data files...")
            df_raw = pd.read_csv('raw_features_data.csv')
            df_unified = pd.read_csv('unified_features_data.csv')
            
            print(f"Loaded raw features data: {len(df_raw)} patients")
            print(f"Loaded unified features data: {len(df_unified)} patients")
            
            # Re-run clustering analysis on loaded data
            raw_results = perform_kmeans_analysis_raw(df_raw, k=4)
            unified_results = perform_kmeans_analysis_unified(df_unified, k=4)
            
            if generate_all:
                # Generate all combinations
                generate_all_visualizations(df_raw, df_unified, raw_results, unified_results)
            else:
                # Original behavior - generate main variations
                # Create visualization
                print("\n" + "="*50)
                print("CREATING COMPARATIVE VISUALIZATION")
                print("="*50)
                
                # 1. Original cluster-based coloring
                print("1. Creating visualization with K-means cluster coloring...")
                visualize_comparative_results(df_raw, df_unified, raw_results, unified_results, 
                                            show_accuracy_borders=True, plot_first_four_only=True, 
                                            color_by_true_groups=False)
                
                # 2. True group coloring with k-means cluster symbols
                print("2. Creating visualization with true group coloring and K-means symbols...")
                visualize_comparative_results(df_raw, df_unified, raw_results, unified_results, 
                                            show_accuracy_borders=True, plot_first_four_only=True, 
                                            color_by_true_groups=True, use_symbols_for_clusters=True)
                
                # 3. True group coloring with color variations (no symbols)
                print("3. Creating visualization with true group coloring and cluster color variations...")
                visualize_comparative_results(df_raw, df_unified, raw_results, unified_results, 
                                            show_accuracy_borders=True, plot_first_four_only=True, 
                                            color_by_true_groups=True, use_symbols_for_clusters=False)
                
                # 4. With decision boundaries (can be messy!)
                print("4. Creating visualization with decision boundaries (experimental)...")
                visualize_comparative_results(df_raw, df_unified, raw_results, unified_results, 
                                            show_accuracy_borders=True, plot_first_four_only=True, 
                                            color_by_true_groups=True, use_symbols_for_clusters=True,
                                            draw_cluster_boundaries=True)
                
                # 5. Minimal version with only PCA scatter and confusion matrix
                print("5. Creating minimal visualization (PCA + Confusion Matrix only)...")
                visualize_comparative_results(df_raw, df_unified, raw_results, unified_results, 
                                            show_accuracy_borders=True, plot_first_four_only=True, 
                                            color_by_true_groups=False, show_pie_chart=False, show_boxplot=False)
                
                # 5b. Minimal version with clean titles
                print("5b. Creating minimal visualization with clean titles...")
                visualize_comparative_results(df_raw, df_unified, raw_results, unified_results, 
                                            show_accuracy_borders=True, plot_first_four_only=True, 
                                            color_by_true_groups=False, show_pie_chart=False, show_boxplot=False,
                                            minimal_titles=True)
                
                # 6. Descriptive only version with only pie chart and boxplot
                print("6. Creating descriptive visualization (Pie Chart + Boxplot only)...")
                visualize_comparative_results(df_raw, df_unified, raw_results, unified_results, 
                                            show_accuracy_borders=True, plot_first_four_only=True, 
                                            color_by_true_groups=False, show_pie_chart=True, show_boxplot=True,
                                            show_clustering_plots=False)
                
                print("\nVISUALIZATION complete!")
                print("Generated files:")
                print("  - 'comparative_kmeans_analysis_4plots.png' for K-means cluster coloring")
                print("  - 'comparative_kmeans_analysis_4plots_true_groups_symbols.png' for true groups + symbols")
                print("  - 'comparative_kmeans_analysis_4plots_true_groups_colors.png' for true groups + color variations")
                print("  - 'comparative_kmeans_analysis_4plots_true_groups_symbols_boundaries.png' for true groups + boundaries")
                print("  - 'comparative_kmeans_analysis_4plots_minimal.png' for PCA + confusion matrix only (with decision boundaries)")
                print("  - 'comparative_kmeans_analysis_4plots_minimal_clean.png' for PCA + confusion matrix with minimal titles")
                print("  - 'comparative_kmeans_analysis_4plots_descriptive_only.png' for pie chart + boxplot only")
                print("\nVisualization Options:")
                print("  • show_pie_chart=False: Removes pathology group distribution pie chart")
                print("  • show_boxplot=False: Removes mean intensity & volume boxplot")
                print("  • show_clustering_plots=False: Removes PCA scatter plots and confusion matrices")
                print("  • show_pie_chart=True, show_boxplot=True, show_clustering_plots=False: Only descriptive plots")
                print("  • color_by_true_groups=True: Colors represent true pathology groups")
                print("  • use_symbols_for_clusters=True: Different symbols for K-means clusters")
                print("  • minimal_titles=True: Use clean, minimal titles for better layout")
                print("  • Decision boundaries automatically enabled for minimal clustering plots")
    
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
    
    # STEP 1: Process patients (raw features only)
    print("\n" + "="*50)
    print("STEP 1: RAW IMAGE FEATURES ANALYSIS")
    print("="*50)
    df_raw = process_patients_raw_only(DATA_PATH, JSON_PATH)
    
    if len(df_raw) == 0:
        print("Error: No raw data processed!")
        return
    
    raw_results = perform_kmeans_analysis_raw(df_raw, k=4)
    
    # STEP 2: Process patients (unified features)
    print("\n" + "="*50)
    print("STEP 2: UNIFIED FEATURES ANALYSIS")
    print("="*50)
    df_unified = process_patients_unified(DATA_PATH, JSON_PATH)
    
    if len(df_unified) == 0:
        print("Error: No unified data processed!")
        return
    
    unified_results = perform_kmeans_analysis_unified(df_unified, k=4)
    
    # 3. Create comparative visualization (6 subplots)
    print("\n" + "="*50)
    print("CREATING COMPARATIVE VISUALIZATION")
    print("="*50)
    
    # 1. Original cluster-based coloring
    print("1. Creating 6-plot visualization with K-means cluster coloring...")
    visualize_comparative_results(df_raw, df_unified, raw_results, unified_results, 
                                show_accuracy_borders=True, plot_first_four_only=False, 
                                color_by_true_groups=False)
    
    # 2. True group coloring with symbols
    print("2. Creating 6-plot visualization with true group coloring and K-means symbols...")
    visualize_comparative_results(df_raw, df_unified, raw_results, unified_results, 
                                show_accuracy_borders=True, plot_first_four_only=False, 
                                color_by_true_groups=True, use_symbols_for_clusters=True)
    
    # 4. Generate comprehensive report
    generate_unified_report(df_unified, unified_results)
    
    # 5. Save data
    df_raw.to_csv('raw_features_data.csv', index=False)
    df_unified.to_csv('unified_features_data.csv', index=False)
    
    print("\nCOMPARATIVE CLUSTERING ANALYSIS complete!")
    print("Check 'comparative_kmeans_analysis.png' for 6-subplot K-means cluster coloring")
    print("Check 'comparative_kmeans_analysis_true_groups.png' for 6-subplot true group coloring")
    print(f"Raw features performance: Silhouette={raw_results['silhouette_score']:.3f}, Accuracy={raw_results['cluster_accuracy']:.3f}")
    print(f"Unified features performance: Silhouette={unified_results['silhouette_score']:.3f}, Accuracy={unified_results['cluster_accuracy']:.3f}")
    
    return df_raw, df_unified, raw_results, unified_results

if __name__ == "__main__":
    # Set skip_processing=True to load existing CSV files and just create plots
    # Set skip_processing=False to reprocess all images (takes longer)
    # Set generate_all=True to create ALL possible visualization combinations
    
    SKIP_PROCESSING = True  # Change to False if you want to reprocess images
    GENERATE_ALL = True     # Change to False for just main variations
    
    df_raw, df_unified, raw_results, unified_results = main(skip_processing=SKIP_PROCESSING, generate_all=GENERATE_ALL)