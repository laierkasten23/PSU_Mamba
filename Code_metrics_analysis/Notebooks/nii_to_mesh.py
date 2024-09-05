import SimpleITK as sitk
import numpy as np
import trimesh
import os

def nii_to_mesh(input_file, output_file, smoothing_iterations=10, smoothing_relaxation=0.1):
    # Load NIfTI image
    img = sitk.ReadImage(input_file)

    # Apply binary thresholding to segment the image
    binary_img = sitk.BinaryThreshold(img, lowerThreshold=0.5, upperThreshold=255)

    # Extract the surface mesh
    mesh = sitk.BinaryMaskToMesh(binary_img)

    # Convert to a numpy array
    vertices = np.array(mesh.GetPoints())
    faces = np.array(mesh.GetPolygons()).reshape(-1, 3)

    # Create a Trimesh object for saving
    surface_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Apply optional smoothing (Laplacian smoothing)
    surface_mesh = surface_mesh.smoothed(smoothing_iterations, relaxation=smoothing_relaxation)

    # Save the mesh
    surface_mesh.export(output_file)

def batch_convert_nii_to_mesh(input_folder, output_folder, output_format="obj"):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".nii") or filename.endswith(".nii.gz"):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + f".{output_format}"
            output_path = os.path.join(output_folder, output_filename)
            nii_to_mesh(input_path, output_path)
            print(f"Converted {filename} to {output_format} and saved at {output_path}")

# Example usage
input_folder = "path/to/your/nifti/folder"
output_folder = "path/to/save/mesh/files"
output_format = "obj"  # Change to "ply" if needed
batch_convert_nii_to_mesh(input_folder, output_folder, output_format)
