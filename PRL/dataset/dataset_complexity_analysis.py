"""
Dataset Complexity Analysis for Choroid Plexus Segmentation

This script analyzes the complexity of a brain MRI dataset containing 4 pathology groups:
- AD: Alzheimer's Disease
- MCI: Mild Cognitive Impairment  
- Psy: Subjective Memory Complaints (Psychiatric)
- other: Other conditions

The analysis includes:
1. K-means clustering on image features
2. Texture analysis 
3. Statistical measures of segmentation complexity
4. Radiomics feature extraction
5. Visualization of results


"""

import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy import ndimage
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def calculate_cluster_accuracy(true_groups, cluster_labels):
    """
    Calculate cluster purity and accuracy metrics
    """
    import pandas as pd
    from collections import Counter
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
    return overall_purity, best_accuracy, best_mapping

class DatasetComplexityAnalyzer:
    def __init__(self, data_path, json_path):
        """
        Initialize the dataset complexity analyzer
        
        Args:
            data_path: Path to patient data folder
            json_path: Path to patients.json file containing group mappings
        """
        self.data_path = data_path
        self.json_path = json_path
        self.patient_groups = self._load_patient_groups()
        self.features_df = None
        self.image_features = []
        self.mask_features = []
        self.patient_ids = []
        self.groups = []
        
    def _load_patient_groups(self):
        """Load patient group mappings from JSON file"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        return data['index_order_all']
    
    def _extract_texture_features(self, image_data):
        """
        Extract texture features from image data using Gray Level Co-occurrence Matrix (GLCM)
        and other statistical measures
        """
        features = {}
        
        # Basic statistical features
        features['mean_intensity'] = np.mean(image_data)
        features['std_intensity'] = np.std(image_data)
        features['skewness'] = self._calculate_skewness(image_data)
        features['kurtosis'] = self._calculate_kurtosis(image_data)
        features['entropy'] = entropy(np.histogram(image_data, bins=256)[0] + 1e-10)
        
        # Gradient-based features
        grad_x = ndimage.sobel(image_data, axis=0)
        grad_y = ndimage.sobel(image_data, axis=1) 
        grad_z = ndimage.sobel(image_data, axis=2)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        
        # Laplacian (edge detection)
        laplacian = ndimage.laplace(image_data)
        features['laplacian_var'] = np.var(laplacian)
        
        return features
    
    def _extract_mask_features(self, mask_data):
        """Extract features from segmentation mask"""
        features = {}
        
        # Volume and shape features
        volume = np.sum(mask_data > 0)
        features['volume'] = volume
        
        if volume > 0:
            # Surface area approximation
            edges = ndimage.sobel(mask_data.astype(float))
            surface_area = np.sum(edges > 0)
            features['surface_area'] = surface_area
            features['surface_to_volume_ratio'] = surface_area / volume if volume > 0 else 0
            
            # Compactness
            features['compactness'] = (surface_area ** 3) / (volume ** 2) if volume > 0 else 0
            
            # Connected components
            labeled, num_components = ndimage.label(mask_data > 0)
            features['num_components'] = num_components
            
            # Centroid and moments
            center_of_mass = ndimage.center_of_mass(mask_data)
            features['centroid_x'] = center_of_mass[0] if not np.isnan(center_of_mass[0]) else 0
            features['centroid_y'] = center_of_mass[1] if not np.isnan(center_of_mass[1]) else 0
            features['centroid_z'] = center_of_mass[2] if not np.isnan(center_of_mass[2]) else 0
        else:
            # Handle empty masks
            features.update({
                'surface_area': 0,
                'surface_to_volume_ratio': 0,
                'compactness': 0,
                'num_components': 0,
                'centroid_x': 0,
                'centroid_y': 0,
                'centroid_z': 0
            })
            
        return features
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def process_all_patients(self):
        """Process all patients and extract features"""
        print("Processing all patients...")
        
        for patient_id in tqdm(range(1, 105)):  # 001 to 104
            patient_folder = f"{patient_id:03d}"
            patient_path = os.path.join(self.data_path, patient_folder)
            
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
                image_features = self._extract_texture_features(image_data)
                mask_features = self._extract_mask_features(mask_data)
                
                # Combine features
                combined_features = {**image_features, **mask_features}
                
                # Store results
                self.image_features.append(image_features)
                self.mask_features.append(mask_features)
                self.patient_ids.append(patient_id)
                self.groups.append(self.patient_groups.get(str(patient_id), 'unknown'))
                
            except Exception as e:
                print(f"Error processing patient {patient_folder}: {e}")
                continue
        
        # Create features DataFrame
        all_features = []
        for i in range(len(self.patient_ids)):
            features = {**self.image_features[i], **self.mask_features[i]}
            features['patient_id'] = self.patient_ids[i]
            features['group'] = self.groups[i]
            all_features.append(features)
            
        self.features_df = pd.DataFrame(all_features)
        print(f"Processed {len(self.features_df)} patients successfully")
        
    def perform_clustering_analysis(self, n_clusters=4):
        """Perform K-means clustering analysis"""
        print(f"Performing K-means clustering with k={n_clusters}...")
        
        # Prepare features for clustering (exclude patient_id and group)
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['patient_id', 'group']]
        X = self.features_df[feature_cols].values
        
        # Handle NaN values
        X = np.nan_to_num(X)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Add cluster labels to DataFrame
        self.features_df['cluster'] = cluster_labels
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        
        # Compare clusters with actual groups
        group_mapping = {'AD': 0, 'MCI': 1, 'Psy': 2, 'other': 3}
        true_labels = [group_mapping.get(group, 3) for group in self.groups]
        ari_score = adjusted_rand_score(true_labels, cluster_labels)
        
        print(f"Silhouette Score: {silhouette_avg:.3f}")
        print(f"Adjusted Rand Index: {ari_score:.3f}")
        
        return {
            'silhouette_score': silhouette_avg,
            'ari_score': ari_score,
            'cluster_labels': cluster_labels,
            'scaler': scaler,
            'kmeans': kmeans
        }
    
    def analyze_group_complexity(self):
        """Analyze complexity differences between groups"""
        print("Analyzing complexity differences between groups...")
        
        complexity_metrics = {}
        
        for group in ['AD', 'MCI', 'Psy', 'other']:
            group_data = self.features_df[self.features_df['group'] == group]
            
            if len(group_data) == 0:
                continue
                
            metrics = {
                'count': len(group_data),
                'mean_volume': group_data['volume'].mean(),
                'std_volume': group_data['volume'].std(),
                'mean_surface_area': group_data['surface_area'].mean(),
                'mean_compactness': group_data['compactness'].mean(),
                'mean_num_components': group_data['num_components'].mean(),
                'mean_intensity': group_data['mean_intensity'].mean(),
                'mean_entropy': group_data['entropy'].mean(),
                'mean_gradient': group_data['gradient_mean'].mean(),
            }
            
            complexity_metrics[group] = metrics
            
        return complexity_metrics
    
    def visualize_results(self, clustering_results):
        """Create visualizations of the analysis results"""
        print("Creating visualizations...")
        
        # Create output directory
        output_dir = "complexity_analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Group distribution
        plt.figure(figsize=(10, 6))
        group_counts = self.features_df['group'].value_counts()
        plt.subplot(2, 2, 1)
        group_counts.plot(kind='bar')
        plt.title('Distribution of Pathology Groups')
        plt.xlabel('Group')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 2. Volume distribution by group
        plt.subplot(2, 2, 2)
        sns.boxplot(data=self.features_df, x='group', y='volume')
        plt.title('Choroid Plexus Volume by Group')
        plt.xticks(rotation=45)
        
        # 3. Clustering results
        plt.subplot(2, 2, 3)
        # PCA for visualization
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['patient_id', 'group', 'cluster']]
        X = self.features_df[feature_cols].values
        X = np.nan_to_num(X)
        X_scaled = clustering_results['scaler'].transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Use discrete colors for categorical clusters - highly distinct colors
        cluster_colors = ['#E74C3C', '#2ECC71', '#3498DB', '#F39C12']  # Red, Green, Blue, Orange
        unique_clusters = sorted(self.features_df['cluster'].unique())
        
        for i, cluster in enumerate(unique_clusters):
            mask = self.features_df['cluster'] == cluster
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=cluster_colors[i], label=f'Cluster {cluster}', 
                       alpha=0.7, s=50)
        
        plt.legend(title='Clusters', loc='upper right')
        plt.title('K-means Clusters (PCA projection)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        
        # 4. Complexity by group
        plt.subplot(2, 2, 4)
        complexity_measure = self.features_df.groupby('group')['compactness'].mean()
        complexity_measure.plot(kind='bar')
        plt.title('Mean Compactness by Group')
        plt.xlabel('Group')
        plt.ylabel('Compactness')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'complexity_analysis_overview.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation matrix of features
        plt.figure(figsize=(12, 10))
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['patient_id', 'group', 'cluster']]
        corr_matrix = self.features_df[feature_cols].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_correlations.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, clustering_results, complexity_metrics):
        """Generate a comprehensive analysis report"""
        print("Generating analysis report...")
        
        output_dir = "complexity_analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        report = []
        report.append("=" * 80)
        report.append("DATASET COMPLEXITY ANALYSIS REPORT")
        report.append("Choroid Plexus Segmentation Dataset")
        report.append("=" * 80)
        report.append("")
        
        # Dataset overview
        report.append("DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total patients processed: {len(self.features_df)}")
        report.append("")
        
        for group, count in self.features_df['group'].value_counts().items():
            report.append(f"{group}: {count} patients")
        report.append("")
        
        # Clustering results
        report.append("CLUSTERING ANALYSIS (K-MEANS)")
        report.append("-" * 40)
        report.append(f"Number of clusters: 4")
        report.append(f"Silhouette Score: {clustering_results['silhouette_score']:.3f}")
        report.append(f"Adjusted Rand Index: {clustering_results['ari_score']:.3f}")
        report.append("")
        
        # Interpretation of clustering results
        if clustering_results['silhouette_score'] > 0.5:
            report.append("✓ Good cluster separation - distinct patterns found")
        elif clustering_results['silhouette_score'] > 0.25:
            report.append("~ Moderate cluster separation - some patterns exist")
        else:
            report.append("✗ Poor cluster separation - limited distinct patterns")
        report.append("")
        
        # Group complexity analysis
        report.append("GROUP COMPLEXITY ANALYSIS")
        report.append("-" * 40)
        
        for group, metrics in complexity_metrics.items():
            report.append(f"{group} Group:")
            report.append(f"  Sample size: {metrics['count']}")
            report.append(f"  Mean volume: {metrics['mean_volume']:.0f} voxels")
            report.append(f"  Mean compactness: {metrics['mean_compactness']:.3f}")
            report.append(f"  Mean components: {metrics['mean_num_components']:.1f}")
            report.append(f"  Mean intensity: {metrics['mean_intensity']:.2f}")
            report.append(f"  Mean entropy: {metrics['mean_entropy']:.3f}")
            report.append("")
        
        # Complexity ranking
        volume_std = {group: metrics['std_volume'] for group, metrics in complexity_metrics.items()}
        complexity_ranking = sorted(volume_std.items(), key=lambda x: x[1], reverse=True)
        
        report.append("COMPLEXITY RANKING (by volume variability):")
        for i, (group, std) in enumerate(complexity_ranking, 1):
            report.append(f"{i}. {group} (std: {std:.1f})")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        if clustering_results['ari_score'] < 0.3:
            report.append("• Low agreement between clusters and pathology groups suggests")
            report.append("  that pathology alone may not determine segmentation complexity")
        
        most_complex = complexity_ranking[0][0]
        least_complex = complexity_ranking[-1][0]
        
        report.append(f"• {most_complex} group shows highest variability - may need more careful handling")
        report.append(f"• {least_complex} group shows most consistent patterns")
        report.append("• Consider stratified sampling or group-specific preprocessing")
        report.append("")
        
        # Save report
        with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
            f.write('\n'.join(report))
        
        # Also save features DataFrame
        self.features_df.to_csv(os.path.join(output_dir, 'extracted_features.csv'), index=False)
        
        # Print report
        for line in report:
            print(line)
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive dataset complexity analysis...")
        print("=" * 60)
        
        # 1. Process all patients
        self.process_all_patients()
        
        if self.features_df is None or len(self.features_df) == 0:
            print("Error: No data processed. Please check file paths and formats.")
            return
        
        # 2. Clustering analysis
        clustering_results = self.perform_clustering_analysis(n_clusters=4)
        
        # 3. Group complexity analysis
        complexity_metrics = self.analyze_group_complexity()
        
        # 4. Visualizations
        self.visualize_results(clustering_results)
        
        # 5. Generate report
        self.generate_report(clustering_results, complexity_metrics)
        
        print("\nAnalysis complete! Results saved in 'complexity_analysis_results' folder.")
        
        return {
            'features_df': self.features_df,
            'clustering_results': clustering_results,
            'complexity_metrics': complexity_metrics
        }


def main():
    """Main function to run the analysis"""
    # Configuration
    DATA_PATH = "/mnt/LIA/pazienti"
    JSON_PATH = "/mnt/LIA/pazienti/patients.json"
    
    # Verify paths exist
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data path {DATA_PATH} does not exist!")
        return
    
    if not os.path.exists(JSON_PATH):
        print(f"Error: JSON file {JSON_PATH} does not exist!")
        return
    
    # Create analyzer and run analysis
    analyzer = DatasetComplexityAnalyzer(DATA_PATH, JSON_PATH)
    results = analyzer.run_full_analysis()
    
    return results


if __name__ == "__main__":
    results = main()
