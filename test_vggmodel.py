import keras
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator


# Layer 효율적 관리를 위한 함수 설정
# MaxPool Layer
def MaxPool(models):
    models.add(MaxPooling2D(pool_size=(2, 2)))


# Convolution Layer
def Conv_layer(models, num_maps, dropout, weight_decay, input=None):
    if input != None:
        # 최초 Convolution Layer
        models.add(Conv2D(num_maps, (3, 3), padding="same", input_shape=input,
                          kernel_regularizer=regularizers.l2(weight_decay)))
        models.add(Activation('relu'))
        models.add(BatchNormalization())
        models.add(Dropout(dropout))
    elif dropout == 0:
        # Dropout을 넣지 않을 경우
        models.add(Conv2D(num_maps, (3, 3), padding="same",
                          kernel_regularizer=regularizers.l2(weight_decay)))
        models.add(Activation('relu'))
        models.add(BatchNormalization())
    else:
        # 다른 경우
        models.add(Conv2D(num_maps, (3, 3), padding="same",
                          kernel_regularizer=regularizers.l2(weight_decay)))
        models.add(Activation('relu'))
        models.add(BatchNormalization())
        models.add(Dropout(dropout))


# Fully Connected Layer
def Dense_layer(models, num_maps, dropout, weight_decay, func_activation, BatchNorm=True):
    models.add(Dense(num_maps,
                     kernel_regularizer=regularizers.l2(weight_decay)))
    models.add(Activation(func_activation))
    if BatchNorm == True:
        models.add(BatchNormalization())
    # Dropout 여부
    if dropout == None:
        return
    else:
        models.add(Dropout(dropout))


if __name__ == "__main__":
    # Main 함수 시작
    # GPU 활용을 위한 코드
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    # Data 받기
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 변수 설정
    lr = 0.1
    lr_decay = 1e-6
    lr_drop = 20
    weight_decay = 0.0005
    num_class = 10
    batch_size = 128

    # Overfitting 방지를 위한 전처리 시작
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Data augmentation
    augment = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
    augment.fit(x_train)
    augment_y = ImageDataGenerator()  # for evaluate_generator

    # Data Normalize
    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)

    # label Categorize
    y_train = keras.utils.to_categorical(y_train, num_class)
    y_test = keras.utils.to_categorical(y_test, num_class)

    # learning rate scheduler

    def lr_schedule(epoch):
        return lr * (0.5 ** (epoch//lr_drop))

    # learning rate decay
    modified_lr = keras.callbacks.LearningRateScheduler(lr_schedule)
    # Optimizer 설정
    sgd = optimizers.SGD(learning_rate=lr, decay=lr_decay,
                         momentum=0.9, nesterov=True)
    rms = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=lr_decay)

    # Model Layer 설정 - VGG16 기반
    model = Sequential()
    Conv_layer(models=model, num_maps=64, dropout=0.3,
               weight_decay=weight_decay, input=x_train.shape[1:])
    Conv_layer(models=model, num_maps=64, dropout=0,
               weight_decay=weight_decay)
    MaxPool(models=model)
    Conv_layer(models=model, num_maps=128,
               dropout=0.3, weight_decay=weight_decay)
    Conv_layer(models=model, num_maps=128,
               dropout=0, weight_decay=weight_decay)
    MaxPool(models=model)
    Conv_layer(models=model, num_maps=256,
               dropout=0.3, weight_decay=weight_decay)
    Conv_layer(models=model, num_maps=256,
               dropout=0.3, weight_decay=weight_decay)
    Conv_layer(models=model, num_maps=256,
               dropout=0, weight_decay=weight_decay)
    MaxPool(models=model)
    Conv_layer(models=model, num_maps=512,
               dropout=0.3, weight_decay=weight_decay)
    Conv_layer(models=model, num_maps=512,
               dropout=0.3, weight_decay=weight_decay)
    Conv_layer(models=model, num_maps=512,
               dropout=0, weight_decay=weight_decay)
    MaxPool(models=model)
    Conv_layer(models=model, num_maps=512,
               dropout=0.3, weight_decay=weight_decay)
    Conv_layer(models=model, num_maps=512,
               dropout=0.3, weight_decay=weight_decay)
    Conv_layer(models=model, num_maps=512,
               dropout=0, weight_decay=weight_decay)
    MaxPool(models=model)
    model.add(Flatten())
    Dense_layer(models=model, num_maps=512, dropout=0.5,
                weight_decay=weight_decay, func_activation='relu')
    Dense_layer(models=model, num_maps=num_class, dropout=0,
                weight_decay=weight_decay, BatchNorm=False, func_activation='softmax')
    model.summary()

    # model compile
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])

    # model training
    training = model.fit_generator(augment.flow(x_train, y_train, batch_size=batch_size),
                                   steps_per_epoch=x_train.shape[0] // batch_size,
                                   epochs=100, validation_data=(x_test, y_test),
                                   callbacks=[modified_lr])

    # model evaluation
    score = model.evaluate_generator(augment_y.flow(x_test, y_test), verbose=1)
    score2 = model.evaluate(x_test, y_test, verbose=1)
    print("Accuracy 1: ", score[1])
    print("Accuracy 2: ", score2[1])
