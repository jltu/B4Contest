import os
import SimpleITK as sItk

DATA_DIR = 'train_seg_output/'
GT_DATA_DIR = 'train/'
OUTPUT_DIR = 'preds/preds'
CASE_LIST_PATH = 'test_case_list.txt'
PRED_CASE_LIST_PATH = 'pred_case_list.txt'


def main():
    # Get the list of case IDs
    with open(CASE_LIST_PATH, 'r') as f:
        case_list = [row.strip() for row in f]

    # Run evaluation and store dice scores
    dice_scores = [evaluate(case) for case in case_list]

    # Show Dice indices
    for case, dice in zip(case_list, dice_scores):
        print('{0}:\t{1:.3f}'.format(case, dice))

    overall_mean_dice = sum(dice_scores)/len(dice_scores)
    print('average:\t{:.3f}'.format(overall_mean_dice))


def evaluate(case):
    # Read the prediction image and ground truth image
    itk_pred = sItk.ReadImage(os.path.join(OUTPUT_DIR, case + '.mhd'))
    itk_gt = sItk.ReadImage(os.path.join(GT_DATA_DIR, 'Tumor', case + '.mhd'))
    itk_image = sItk.ReadImage(os.path.join(DATA_DIR, case + '.mhd'))

    # Convert ITK arrays into NumPy boolean arrays
    pred = sItk.GetArrayFromImage(itk_pred).astype(bool)
    gt = sItk.GetArrayFromImage(itk_gt).astype(bool)
    image = sItk.GetArrayFromImage(itk_image).astype(bool)

    # Calculate Dice index inside the mask
    return calc_dice(pred[image], gt[image])


def calc_dice(a, b):
    # Compute Dice coefficient
    return 2. * (a & b).sum() / (a.sum() + b.sum())


if __name__ == '__main__':
    main()



