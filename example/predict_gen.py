from __future__ import print_function

import SimpleITK as sItk

from data import load_train_data, load_test_data
from train import get_unet
from utils import *

PRED_DIR = r'C:\Users\jonathantu\B4Contest\example\preds'


def test_and_predict():

    # Load in the training and mask data
    imgs_train, imgs_mask_train = load_train_data()
    print('imgs_train', type(imgs_train))

    imgs_train = imgs_train.astype('float32')

    imgs_train = imgs_train[..., np.newaxis]

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    print('-' * 30)
    print('Loading and preprocessing test data...')
    print('-' * 30)
    imgs_test, imgs_id_test = load_test_data()

    # Adding a new axis for channel length
    imgs_test = imgs_test.astype('float32')
    imgs_test = imgs_test[..., np.newaxis]
    imgs_test -= mean
    imgs_test /= std

    # Obtain the unet model
    model = get_unet()

    # Load the weights
    print('-' * 30)
    print('Loading saved weights...')
    print('-' * 30)
    model.load_weights('weights.h5')

    print('-' * 30)
    print('Predicting masks on test data...')
    print('-' * 30)
    imgs_mask_test = model.predict(imgs_test, batch_size=2, verbose=1)
    np.save('imgs_tumor_test.npy', imgs_mask_test)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    i = 1
    length = len(imgs_mask_test)
    image_rows = 1024  # height
    image_cols = 512  # width

    # Reshape the mask array to a 3D array from a 4D array
    imgs_mask_test = np.reshape(imgs_mask_test, [length, image_rows, image_cols])
    print('shape after reshape', imgs_mask_test.shape)

    if not os.path.exists(PRED_DIR):
        os.mkdir(PRED_DIR)
    for j in range(len(imgs_mask_test)):
        # EUDT
        preds_image = sItk.GetImageFromArray(imgs_mask_test[j])
        preds_image.SetOrigin([0, 0, 0])
        preds_image.SetSpacing([2.8, 2.8, 1])

        # output image
        write_mhd_and_raw(preds_image, '{}.mhd'.format(os.path.join(PRED_DIR, 'preds', 'preds_{}'.format(j + 1))))

        print('Done: {0}/{1} images'.format(i, length))
        i += 1

    print('-' * 30)
    print('Done...')


if __name__ == '__main__':
    test_and_predict()