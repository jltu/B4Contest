import os
import SimpleITK as sItk

DATA_DIR = 'train/'
OUTPUT_DIR = 'output'
CASE_LIST_PATH = 'test_case_list.txt'


def main():
    # Get the list of case IDs
    with open(CASE_LIST_PATH, 'r') as f:
        case_list = [row.strip() for row in f]

    # Run evaluation and store dice scores
    dice_scores = [evaluate(case) for case in case_list]

    # Show Dice indices
    for case, dice in zip(case_list, dice_scores):
        print('{0}:\t{1:.3f}'.format(case, dice))

    mean_dice = sum(dice_scores)/len(dice_scores)
    print('average:\t{:.3f}'.format(mean_dice))


def evaluate(case):
    # Read the target image and the bone label image
    itk_seg = sItk.ReadImage(os.path.join(OUTPUT_DIR, case + '.mhd'))
    itk_gt = sItk.ReadImage(os.path.join(DATA_DIR, 'Tumor/', case + '.mhd'))
    itk_mask = sItk.ReadImage(os.path.join(DATA_DIR, 'Bone/', case + '.mhd'))

    # Convert ITK arrays into NumPy boolean arrays
    seg = sItk.GetArrayFromImage(itk_seg).astype(bool)
    gt = sItk.GetArrayFromImage(itk_gt).astype(bool)
    mask = sItk.GetArrayFromImage(itk_mask).astype(bool)

    # Calculate Dice index inside the mask
    return calc_dice(seg[mask], gt[mask])


def calc_dice(a, b):
    # Compute Dice coefficient
    return 2. * (a & b).sum() / (a.sum() + b.sum())


if __name__ == '__main__':
    main()
