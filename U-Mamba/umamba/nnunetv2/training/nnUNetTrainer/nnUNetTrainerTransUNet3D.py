# nnunetv2/training/nnUNetTrainer/nnUNetTrainerTransUNet3D.py
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.nets.TransUnet3d import TransUNet3D  # adjust path if different

class nnUNetTrainerTransUNet3D(nnUNetTrainer):
    # ViT patching; must divide planner patch size along (D, H, W)
    vit_patch_size = (2, 16, 16)

    # We return a single output head; ignore DS from v2 for this model
    enable_deep_supervision = False

    def build_network_architecture(
        self,
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels,
        enable_deep_supervision
    ):
        # Get planner patch size (D, H, W)
        ps = tuple(int(x) for x in configuration_manager.patch_size)  # e.g., (32, 192, 192)
        req = self.vit_patch_size
        for a, b in zip(ps, req):
            if a % b != 0:
                raise ValueError(
                    f"Planner patch_size {ps} must be divisible by vit_patch_size {req} for ViT-3D patches."
                )

        net = TransUNet3D(
            in_channels=num_input_channels,
            num_classes=self.label_manager.num_segmentation_heads,
            img_size=ps,
            patch_size=req,
            embed_dim=384,    # bump to 512/768 if VRAM allows
            depth=8,
            num_heads=6,
            mlp_dim=1536,
            drop=0.0,
            decoder_channels=(256, 128, 64, 32),
        )
        return net
