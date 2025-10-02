"""
Detailed cluster accuracy analysis
"""

import pandas as pd
import numpy as np

def analyze_cluster_accuracy():
    # Load the results
    df = pd.read_csv('features_data.csv')
    
    print("=== DETAILED CLUSTER ACCURACY ANALYSIS ===")
    print()
    
    # Create confusion matrix
    confusion_matrix = pd.crosstab(df['cluster'], df['group'], margins=True)
    print("Confusion Matrix (Cluster vs True Group):")
    print(confusion_matrix)
    print()
    
    # Calculate accuracy for each cluster
    print("Cluster-wise Accuracy:")
    print("-" * 40)
    
    cluster_accuracies = {}
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        
        # Find most common group in this cluster
        most_common_group = cluster_data['group'].value_counts().index[0]
        correct_predictions = len(cluster_data[cluster_data['group'] == most_common_group])
        total_in_cluster = len(cluster_data)
        
        accuracy = correct_predictions / total_in_cluster
        cluster_accuracies[cluster_id] = accuracy
        
        print(f"Cluster {cluster_id}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  → Best match: {most_common_group} ({correct_predictions}/{total_in_cluster} patients)")
        
        # Show breakdown
        group_counts = cluster_data['group'].value_counts()
        print("  → Group breakdown:", end=" ")
        for group, count in group_counts.items():
            pct = (count / total_in_cluster) * 100
            print(f"{group}:{count}({pct:.1f}%)", end=" ")
        print()
        print()
    
    # Overall statistics
    total_patients = len(df)
    weighted_accuracy = sum(cluster_accuracies[cid] * len(df[df['cluster'] == cid]) 
                           for cid in cluster_accuracies) / total_patients
    
    print("SUMMARY:")
    print(f"• Total patients: {total_patients}")
    print(f"• Weighted cluster purity: {weighted_accuracy:.3f} ({weighted_accuracy*100:.1f}%)")
    print(f"• This means {weighted_accuracy*100:.1f}% of patients are in the 'correct' cluster")
    print(f"• Random chance would be ~{100/4:.1f}% (25.0%)")
    
    # Improvement over random
    improvement = (weighted_accuracy - 0.25) / 0.25 * 100
    if improvement > 0:
        print(f"• Clustering is {improvement:.1f}% better than random assignment")
    else:
        print(f"• Clustering performs {abs(improvement):.1f}% worse than random assignment")

if __name__ == "__main__":
    analyze_cluster_accuracy()
