import os

def create_nnunet_dataset_from_bids(
    path,
    task_id,
    task_name,
    output_dir,
    image_pattern="sub-{sid}_ses-001_INV1.nii.gz",
    seg_pattern="sub-{sid}_ses-001_training_labels_brainstem.nii.gz",
    train_subjects=None,
    val_subjects=None,
    test_subjects=None,
    modality_identifier="0000",
    file_ending=".nii.gz"
):
    """
    Create a nnU-Net style dataset from a BIDS folder structure using symbolic links.

    :param path: Path to the BIDS dataset (contains sub-00x folders)
    :param task_id: Integer task ID (e.g., 33)
    :param task_name: Name of the task (e.g., "ChoroidPlexus")
    :param output_dir: Where to create the nnUNet dataset (e.g., /path/to/nnUNet_raw)
    :param image_pattern: Pattern for the image file in anat/ (default: sub-{sid}_ses-001_INV1.nii.gz)
    :param seg_pattern: Pattern for the seg file in seg/ (default: sub-{sid}_ses-001_training_labels_brainstem.nii.gz)
    :param train_subjects: List of subject IDs for training (e.g., ["001", "002"])
    :param val_subjects: List of subject IDs for validation (optional)
    :param test_subjects: List of subject IDs for testing (optional)
    :param modality_identifier: 4-digit string for the modality/channel (default: "0000")
    :param file_ending: File ending for images and labels (default: ".nii.gz")
    """
    dataset_name = f"Dataset{task_id:03d}_{task_name}"
    dataset_path = os.path.join(output_dir, dataset_name)
    imagesTr = os.path.join(dataset_path, "imagesTr")
    imagesVal = os.path.join(dataset_path, "imagesVal")
    imagesTs = os.path.join(dataset_path, "imagesTs")
    labelsTr = os.path.join(dataset_path, "labelsTr")
    labelsVal = os.path.join(dataset_path, "labelsVal")

    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    if val_subjects:
        os.makedirs(imagesVal, exist_ok=True)
        os.makedirs(labelsVal, exist_ok=True)
    if test_subjects:
        os.makedirs(imagesTs, exist_ok=True)
        
    # If no split is specified, use all subjects as training
    if train_subjects is None and val_subjects is None and test_subjects is None:
        all_subjects = sorted([
            d.replace("sub-", "") for d in os.listdir(path)
            if d.startswith("sub-") and os.path.isdir(os.path.join(path, d))
        ])
        train_subjects = all_subjects
        val_subjects = []
        test_subjects = []
        print("No train/val/test split specified: using all subjects as training set.")


    def symlink_case(sid, split):
        # CASE_IDENTIFIER for nnUNet: MRI_IDsj = sub-00x
        case_id = f"sub-{sid}"
        case_id_dest = f"{sid}"
        img_src = os.path.join(path, case_id, "anat", image_pattern.format(sid=sid))
        seg_src = os.path.join(path, case_id, "seg", seg_pattern.format(sid=sid))
        img_dst = None
        seg_dst = None

        if split == "train":
            img_dst = os.path.join(imagesTr, f"{case_id_dest}_{modality_identifier}{file_ending}")
            seg_dst = os.path.join(labelsTr, f"{case_id_dest}{file_ending}")
        elif split == "val":
            img_dst = os.path.join(imagesVal, f"{case_id_dest}_{modality_identifier}{file_ending}")
            seg_dst = os.path.join(labelsVal, f"{case_id_dest}{file_ending}")
        elif split == "test":
            img_dst = os.path.join(imagesTs, f"{case_id_dest}_{modality_identifier}{file_ending}")
            # No seg_dst for test

        # Create symlinks if not already present
        if not os.path.exists(img_dst):
            os.symlink(img_src, img_dst)
            print(f"Linked image: {img_src} -> {img_dst}")
        if seg_dst and not os.path.exists(seg_dst):
            if os.path.exists(seg_src):
                os.symlink(seg_src, seg_dst)
                print(f"Linked seg: {seg_src} -> {seg_dst}")

    # Link training cases
    if train_subjects:
        for sid in train_subjects:
            symlink_case(sid, "train")
    # Link validation cases
    if val_subjects:
        for sid in val_subjects:
            symlink_case(sid, "val")
    # Link test cases
    if test_subjects:
        for sid in test_subjects:
            symlink_case(sid, "test")
            

create_nnunet_dataset_from_bids(
    path="/data1/user/BIDS",
    task_id=299,
    task_name="BrainStem",
    output_dir="/data1/user/Umamba_data/nnUNet_raw",
    image_pattern="sub-{sid}_ses-001_INV1.nii.gz",
    seg_pattern="sub-{sid}_ses-001_training_labels_brainstem.nii.gz",
    modality_identifier="0001",
    file_ending=".nii.gz"
)