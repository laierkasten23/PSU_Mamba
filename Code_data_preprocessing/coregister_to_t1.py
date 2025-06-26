import nibabel as nib
import SimpleITK as sitk
import os

parent_dir = "/home/linuxuser/user/data/own_coregistration_240831/022"
# Load the T1 (fixed) and FLAIR (moving) images using nibabel
fixed_image = sitk.ReadImage(os.path.join(parent_dir,'022_T1.nii'), sitk.sitkFloat32)
moving_image = sitk.ReadImage(os.path.join(parent_dir, '022_ChP_mask_T1xFLAIR_manual_seg.nii'), sitk.sitkFloat32)

# Initialize the registration method
registration_method = sitk.ImageRegistrationMethod()

# Similarity metric: Normalized Mutual Information (NMI)
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

# Interpolator: Trilinear interpolation
registration_method.SetInterpolator(sitk.sitkLinear)

# Optimizer: Gradient descent with default settings
registration_method.SetOptimizerAsRegularStepGradientDescent(
    learningRate=2.0, 
    minStep=1e-4, 
    numberOfIterations=200, 
    gradientMagnitudeTolerance=1e-8, 
    relaxationFactor=0.5
)

# Initial transform: Centered transform initializer (affine)
initial_transform = sitk.CenteredTransformInitializer(
    fixed_image, 
    moving_image, 
    sitk.AffineTransform(fixed_image.GetDimension()), 
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)
registration_method.SetInitialTransform(initial_transform, inPlace=False)

# Set the scale of the translation components relative to the other parameters
registration_method.SetOptimizerScalesFromPhysicalShift()

# Set a smaller rotation and translation search space
registration_method.SetOptimizerAsRegularStepGradientDescent(
    learningRate=1.0, 
    minStep=1e-4, 
    numberOfIterations=200
)

# Execute the registration
final_transform = registration_method.Execute(fixed_image, moving_image)

# Resample the moving image (FLAIR) onto the fixed image (T1-weighted)
resampled_image = sitk.Resample(
    moving_image, 
    fixed_image, 
    final_transform, 
    sitk.sitkLinear, 
    0.0, 
    moving_image.GetPixelID()
)

# Define full path to save the registered image
output_path = os.path.join(parent_dir, '022_ChP_mask_T1xFLAIR_manual_seg_new.nii')

# Save the registered (resampled) image
sitk.WriteImage(resampled_image, output_path)
print("Coregistration complete and image saved to %s" % output_path)
