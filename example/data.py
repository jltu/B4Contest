from __future__ import print_function

import os
import numpy as np
import SimpleITK as sItk
import re

SEG_PATH_TRAIN = 'output/'
DATA_PATH_TRAIN = 'train/'
DATA_PATH_TEST = 'test_output/'
CASE_LIST_PATH = 'test_case_list.txt'

image_rows = 1024       # height
image_cols = 512        # width


# ------------------------------------------------------------------------------------------------
# Train Data


def create_img_train_data():
    train_data_path = SEG_PATH_TRAIN
    images = os.listdir(train_data_path)
    num = len(images)
    count = num / 2

    imgs = np.ndarray((int(count), int(image_rows), int(image_cols)), dtype=np.uint8)

    i = 0
    print('-' * 30)
    print('Creating training images for Image...')
    print('-' * 30)

    for filename in images:
        if filename.endswith('.mhd'):
            # Read the target segmented image
            itk_img = sItk.ReadImage(os.path.join(SEG_PATH_TRAIN, filename))
            print(filename)
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
    print("imgs_train",imgs.shape,imgs)

    np.save('imgs_train.npy', imgs)
    print('Saving to .npy files done.')


def create_mask_train_data():
    train_data_path = os.path.join(DATA_PATH_TRAIN, 'Tumor')
    mask_images = os.listdir(train_data_path)
    num = len(mask_images)
    count = num / 2

    imgs_mask = np.ndarray((int(count), int(image_rows), int(image_cols)), dtype=np.uint8)

    i = 0
    print('-' * 30)
    print('Creating Ground Truth Images...')
    print('-' * 30)

    for filename in mask_images:
        if filename.endswith('.mhd'):
            # Read the target image and the bone label image
            itk_img_mask = sItk.ReadImage(os.path.join(DATA_PATH_TRAIN, 'Tumor/', filename))
            print(filename)
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
    print("imgs_mask",imgs_mask.shape,imgs_mask)

    np.save('imgs_mask_train.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


# ------------------------------------------------------------------------------------------------
# Test Data


def create_img_test_data():
    train_data_path = DATA_PATH_TEST
    images = os.listdir(train_data_path)
    num = len(images)
    count = num / 2

    imgs = np.ndarray((int(count), int(image_rows), int(image_cols)), dtype=np.uint8)
    imgs_id = np.ndarray((int(count), ), dtype=np.int32)


    i = 0
    print('-' * 30)
    print('Creating segmented images for test...')
    print('-' * 30)

    for filename in images:
        if filename.endswith('.mhd'):
            # Read the target image and the bone label image
            itk_img = sItk.ReadImage(os.path.join(DATA_PATH_TEST, filename))
            img_id = int(re.search(r'\d+', filename).group())
            print(filename)
            # Convert ITK arrays into NumPy array
            img = sItk.GetArrayFromImage(itk_img)

            img = np.array([img])
            imgs[i] = img

            imgs_id[i] = img_id

            print('Done: {0}/{1} images'.format(i, count))

            i += 1
            continue
        else:
            continue

    print('Loading done.')
    print("imgs_test", imgs.shape, imgs)

    np.save('imgs_test.npy', imgs)
    np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id


if __name__ == '__main__':
    create_img_train_data()
    create_mask_train_data()
    create_img_test_data()
