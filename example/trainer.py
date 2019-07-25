from __future__ import print_function

import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from data import load_train_data

K.set_image_data_format('channels_last')

img_rows = 1024
img_cols = 512

smooth = 0.0000001


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def get_unet():
    inputs = Input((img_rows, img_cols, 1))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

    # Commented out to simplify network due to hardware limitations
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    # conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    # up6 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
    #                    (conv4), conv3], axis=3)
    # conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    # conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
                       (conv4), conv3], axis=3)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')
                       (conv7), conv2], axis=3)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')
                       (conv8), conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

    return model


def train():
    # Load training and mask data
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_mask_train = imgs_mask_train.astype('uint8')
    imgs_train = imgs_train.astype('float32')

    # Add a new axis to change from a 3D array into a 4D array
    imgs_mask_train = imgs_mask_train[..., np.newaxis]
    imgs_train = imgs_train[..., np.newaxis]

    print("imgs_train", imgs_train.shape)

    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')

    # Create and compile the model and saving it to HDF5 file
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet()
    model_checkpoint = ModelCheckpoint('weights2.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    # Debug statements to confirm output from training data
    # inputs=np.reshape(imgs_train[0], [1024,512])
    # inputs=sItk.GetImageFromArray(inputs)
    # inputs.SetSpacing([2.8,2.8])
    # sItk.WriteImage(inputs, r'C:\Users\jonathantu\B4Contest\example\confirm_input\CHECK.mhd', True)
    # mask_inputs = np.reshape(imgs_mask_train[0], [1024, 512])
    # mask_inputs=sItk.GetImageFromArray(mask_inputs)
    # mask_inputs.SetSpacing([2.8,2.8])
    # sItk.WriteImage(mask_inputs, r'C:\Users\jonathantu\B4Contest\example\confirm_input\CHECK_mask.mhd')

    # Fitting the model
    model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=20, verbose=1, shuffle=True,
              validation_split=0.2,
              callbacks=[model_checkpoint])

    print('-' * 30)
    print('Saving weights...')
    print('-' * 30)


if __name__ == '__main__':
    train()
