import os
import numpy as np
import SimpleITK as sItk

DATA_DIR = 'train/'
OUTPUT_DIR = 'train_seg_output'
CASE_LIST_PATH = 'test_case_list.txt'
MASK_OUTPUT_DIR = 'Debug_Mask_Output'


def main():
    # Get the list of case IDs
    with open(CASE_LIST_PATH, 'r') as f:
        case_list = [row.strip() for row in f]

    # Run segmentation process for all cases
    for case in case_list:
        run_segmentation(case)


def run_segmentation(case):
    # Read the target image and the bone label image
    itk_img = sItk.ReadImage(os.path.join(DATA_DIR, 'Image/', case + '.mhd'))
    itk_mask = sItk.ReadImage(os.path.join(DATA_DIR, 'Bone/', case + '.mhd'))

    # Convert ITK arrays into NumPy array
    img = sItk.GetArrayFromImage(itk_img)
    mask = sItk.GetArrayFromImage(itk_mask)

    # Set all values in mask to be 1
    mask[mask > 0] = 1

    # Apply Element-wise product to image and mask
    # Outside of the bone should be masked out
    seg = np.multiply(img, mask)

    # seg = seg.astype(np.uint8)

    # Convert a NumPy array into an ITK array
    itk_seg = sItk.GetImageFromArray(seg)
    # Copy image information (e.g., image spacing)
    itk_seg.CopyInformation(itk_img)

    # Write segmentation result
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # seg_image = sItk.GetImageFromArray(itk_seg)
    itk_seg.SetOrigin([0, 0, 0])
    itk_seg.SetSpacing([2.8, 2.8, 1])
    sItk.WriteImage(itk_seg, os.path.join(OUTPUT_DIR, case + '.mhd'))

    ##################################################################################
    # # DEBUG code to check mask
    # # Convert a NumPy array into an ITK array
    # itk_mask = sItk.GetImageFromArray(mask)
    # # Copy image information (e.g., image spacing)
    # itk_mask.CopyInformation(itk_mask)
    # if not os.path.exists(MASK_OUTPUT_DIR):
    #     os.makedirs(MASK_OUTPUT_DIR)
    # itk_mask.SetOrigin([0, 0, 0])
    # itk_mask.SetSpacing([2.8, 2.8, 1])
    # sItk.WriteImage(itk_mask, os.path.join(MASK_OUTPUT_DIR, case + '.mhd'))


if __name__ == '__main__':
    main()
