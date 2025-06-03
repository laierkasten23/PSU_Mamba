import SimpleITK as sitk

def load_image(file_path):
    """
    Load a NIfTI image using SimpleITK.
    
    Args:
    - file_path (str): Path to the NIfTI file.
    
    Returns:
    - SimpleITK.Image: Loaded image.
    """
    return sitk.ReadImage(file_path)

def register_images(fixed_image, moving_image):
    """
    Register the moving image to the fixed image using SimpleITK.
    
    Args:
    - fixed_image (SimpleITK.Image): The reference image (e.g., T1).
    - moving_image (SimpleITK.Image): The image to be registered (e.g., FLAIR).
    
    Returns:
    - SimpleITK.Transform: The resulting transformation.
    """
    # Initialize the registration method
    registration_method = sitk.ImageRegistrationMethod()
    
    # Similarity metric settings
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)
    
    # Interpolator settings
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimizer settings
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Initial transform
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # Multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Execute the registration
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))
    
    return final_transform

def apply_transform(moving_image, transform, reference_image):
    """
    Apply the transformation to the moving image.
    
    Args:
    - moving_image (SimpleITK.Image): The image to be transformed.
    - transform (SimpleITK.Transform): The transformation to apply.
    - reference_image (SimpleITK.Image): The reference image for resampling.
    
    Returns:
    - SimpleITK.Image: The transformed image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)
    
    return resampler.Execute(moving_image)

def save_image(image, file_path):
    """
    Save a SimpleITK image to a NIfTI file.
    
    Args:
    - image (SimpleITK.Image): The image to save.
    - file_path (str): Path to the output NIfTI file.
    """
    sitk.WriteImage(image, file_path)

# Example usage
root_dir = '/home/linuxlia/Lia_Masterthesis/data/pazienti1_52_Lia'
subject_id = '038'
fixed_image_path = f'{root_dir}/{subject_id}/{subject_id}_T1.nii'
moving_image_path = f'{root_dir}/{subject_id}/{subject_id}_FLAIR.nii'
output_image_path = f'{root_dir}/{subject_id}/{subject_id}_FLAIR_Lia_registered.nii'

fixed_image = load_image(fixed_image_path)
moving_image = load_image(moving_image_path)

transform = register_images(fixed_image, moving_image)
print("transform = ", transform)
print(transform.GetParameters())
registered_image = apply_transform(moving_image, transform, fixed_image)

save_image(registered_image, output_image_path)