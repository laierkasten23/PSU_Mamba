import os
import numpy as np
import nibabel as nib

def compute_pca_scan_path(self, dataset: nnUNetDataset, fold: int):
    pred_dir = os.path.join(self.output_folder, "validation_probs")
    all_probs = []
    
    for key in dataset.dataset.keys():
        prob_path = os.path.join(pred_dir, f"{key}_prob.nii.gz")
        if not os.path.exists(prob_path):
            continue
        prob_img = nib.load(prob_path)
        prob_data = prob_img.get_fdata()
        all_probs.append(prob_data)

    mean_prob = np.mean(all_probs, axis=0)
    coords = np.argwhere(mean_prob > 0.1)  # Only keep foreground
    weights = mean_prob[mean_prob > 0.1]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(coords * weights[:, None])
    pc1 = pca.components_[0]

    save_path = os.path.join(self.output_folder, f"pca_scan_path_fold{fold}.npy")
    np.save(save_path, pc1)
    os.chmod(save_path, 0o777)
    print("PCA scan path saved to", save_path)
    return pc1
