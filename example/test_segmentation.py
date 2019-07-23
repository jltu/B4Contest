import os
import numpy as np
import SimpleITK as sItk

DATA_DIR = 'test/'
OUTPUT_DIR = 'test_output'
CASE_LIST_PATH = 'test/case_list.txt'


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
    mask = sItk.GetArrayFromImage(itk_mask).astype(bool)

    # # Apply some simple thresholds
    # # Outside the bone should be masked out
    # seg = (img >= 0) & mask
    # # Output pixel type should be Byte
    # seg = seg.astype(np.uint8)

    # Apply Element wise product to mask out the bone from the image
    # Outside of the bone should be masked out
    seg = np.multiply(img, mask)
    seg = seg.astype(np.uint8)

    # Convert a NumPy array into an ITK array
    itk_seg = sItk.GetImageFromArray(seg)
    # Copy image information (e.g., image spacing)
    itk_seg.CopyInformation(itk_img)

    # Write segmentation result
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    sItk.WriteImage(itk_seg, os.path.join(OUTPUT_DIR, case + '.mhd'))


if __name__ == '__main__':
    main()
