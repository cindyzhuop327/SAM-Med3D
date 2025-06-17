# -*- encoding: utf-8 -*-
'''
@File    :   prepare_data_from_nnUNet.py
@Time    :   2023/12/10 23:07:39
@Author  :   Haoyu Wang 
@Contact :   small_dark@sina.com
@Brief   :   pre-process nnUNet-style dataset into SAM-Med3D-style
'''

import os.path as osp
import os
import json
import shutil
import nibabel as nib
from tqdm import tqdm
import torchio as tio
from glob import glob
import torch

def resample_nii(input_path: str, output_path: str, target_spacing: tuple = (1.5, 1.5, 1.5), n=None, reference_image=None):
    """
    Resample a nii.gz file to a specified spacing using torchio.

    Parameters:
    - input_path: Path to the input .nii.gz file.
    - output_path: Path to save the resampled .nii.gz file.
    - target_spacing: Desired spacing for resampling. Default is (1.5, 1.5, 1.5).
    """
    
    # Load the nii.gz file using torchio
    subject = tio.Subject(
        img=tio.ScalarImage(input_path)
    )

    # Resample the image to the target spacing
    resampler = tio.Resample(target=target_spacing)
    resampled_subject = resampler(subject)
    if(n!=None):
        image = resampled_subject.img
        tensor_data = image.data
        
        # Binarize based on the class index 'n'
        temp_tensor = torch.zeros_like(tensor_data)
        if isinstance(n, int):
            temp_tensor[tensor_data == n] = 1
        elif isinstance(n, list):
            for ni_val in n:
                temp_tensor[tensor_data == ni_val] = 1
        tensor_data = temp_tensor # Now tensor_data is binary for the target classes

        save_image = tio.ScalarImage(tensor=tensor_data, affine=image.affine)
        if reference_image is not None:
            reference_size = reference_image.shape[1:]  # omitting the channel dimension
            cropper_or_padder = tio.CropOrPad(reference_size)
            save_image = cropper_or_padder(save_image)
    else:
        save_image = resampled_subject.img
    
    # Save the resampled image to the specified output path
    save_image.save(output_path)

dataset_root = "/raid/data/nnunet/nnUNET_raw/Dataset029_ACDC/"

# Removed dataset_list as it's no longer used in the same way

target_dir = "/home/zhuox01/SAM-Med3D/data/processed_ACDC/" # Suggesting a new output directory

# Load the dataset.json directly from the dataset_root
meta_info = json.load(open(osp.join(dataset_root, "dataset.json")))

print(meta_info.get('name', 'DatasetNameNotFound')) # Use .get to avoid KeyError for 'name'
num_classes = len(meta_info["labels"])-1
print("num_classes:", num_classes, meta_info["labels"])

# Invert labels dictionary if it's in "name": value format
if all(isinstance(k, str) and not k.isdigit() for k in meta_info["labels"].keys()):
    labels_map = {str(v): k for k, v in meta_info["labels"].items()}
else:
    labels_map = meta_info["labels"]

# Get a list of all image files in imagesTr and extract their base names (patient IDs)
image_files = sorted(glob(osp.join(dataset_root, "imagesTr", "*.nii.gz")))
case_ids = [osp.basename(f).replace("_0000.nii.gz", "").replace(".nii.gz", "") for f in image_files]

dataset_name_for_output = meta_info.get('name', 'ACDC').replace("Dataset000_", "") # Use the name from dataset.json for the output folder

# Create the top-level target directory
os.makedirs(target_dir, exist_ok=True)

for idx_str, cls_name_str in labels_map.items():
    idx = int(idx_str)
    if idx == 0:  # Skip background
        continue
    cls_name = cls_name_str.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

    target_cls_dir = osp.join(target_dir, cls_name, dataset_name_for_output)
    target_img_dir = osp.join(target_cls_dir, "imagesTr")
    target_gt_dir = osp.join(target_cls_dir, "labelsTr")
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_gt_dir, exist_ok=True)

    for case_id in tqdm(case_ids, desc=f"{dataset_name_for_output}-{cls_name}"):
        img_orig_path = osp.join(dataset_root, "imagesTr", f"{case_id}_0000.nii.gz")
        gt_orig_path = osp.join(dataset_root, "labelsTr", f"{case_id}.nii.gz")

        # Resample image and label
        resample_img_path = osp.join(target_img_dir, f"{case_id}.nii.gz") # Save directly to target
        resample_gt_path = osp.join(target_gt_dir, f"{case_id}.nii.gz") # Save directly to target

        # Only resample if not already exists
        if not osp.exists(resample_img_path):
            resample_nii(img_orig_path, resample_img_path)

        gt_img = nib.load(gt_orig_path)
        spacing = tuple(gt_img.header['pixdim'][1:4])
        spacing_voxel = spacing[0] * spacing[1] * spacing[2]
        gt_arr = gt_img.get_fdata()

        # Create binary mask for the current class
        gt_arr_binary = (gt_arr == idx).astype(int) # Use '==' for specific label, then convert to int (0 or 1)

        volume = gt_arr_binary.sum() * spacing_voxel
        if volume < 10:
            print(f"skip {resample_gt_path} (volume < 10)")
            continue

        # Use the resampled image as reference for label resampling (padding/cropping)
        reference_image = tio.ScalarImage(resample_img_path) # Use the already resampled image
        # The original resample_nii uses 'n' for binary conversion, we need to pass 'idx' here
        resample_nii(gt_orig_path, resample_gt_path, n=idx, reference_image=reference_image)

        # Removed shutil.copy(img, target_img_path) as resample_nii now handles saving


