from __future__ import print_function

import os
import numpy as np
import SimpleITK as sItk

data_path = 'train/'
CASE_LIST_PATH = 'test_case_list.txt'

image_rows = 1024
image_cols = 512


def create_img_train_data():
    train_data_path = os.path.join(data_path, 'Image')
    images = os.listdir(train_data_path)
    num = len(images)
    count = num / 2

    imgs = np.ndarray((int(num), int(image_rows), int(image_cols)), dtype=np.uint8)

    i = 1
    print('-' * 30)
    print('Creating training images for Image...')
    print('-' * 30)

    for filename in images:
        if filename.endswith('.mhd'):
            # Read the target image and the bone label image
            itk_img = sItk.ReadImage(os.path.join(data_path, 'Image/', filename))

            # Convert ITK arrays into NumPy array
            img = sItk.GetArrayFromImage(itk_img)

            img = np.array([img])
            imgs[i] = img

            print('Done: {0}/{1} images'.format(i, count))

            i += 1
            continue
        else:
            continue

    print('Loading done.')
    print(imgs[i])

    np.save('imgs_train.npy', imgs)
    print('Saving to .npy files done.')


def create_mask_train_data():
    train_data_path = os.path.join(data_path, 'Bone')
    mask_images = os.listdir(train_data_path)
    num = len(mask_images)
    count = num / 2

    imgs_mask = np.ndarray((int(num), int(image_rows), int(image_cols)), dtype=np.uint8)

    i = 1
    print('-' * 30)
    print('Creating training images for Mask...')
    print('-' * 30)

    for filename in mask_images:
        if filename.endswith('.mhd'):
            # Read the target image and the bone label image
            itk_img_mask = sItk.ReadImage(os.path.join(data_path, 'Bone/', filename))

            # Convert ITK arrays into NumPy array
            img_mask = sItk.GetArrayFromImage(itk_img_mask)

            img_mask = np.array([img_mask])
            imgs_mask[i] = img_mask

            print('Done: {0}/{1} images'.format(i, count))

            i += 1
            continue
        else:
            continue

    print('Loading done.')
    print(imgs_mask[i])

    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


if __name__ == '__main__':
    create_img_train_data()
    create_mask_train_data()
