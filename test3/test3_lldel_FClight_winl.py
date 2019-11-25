import keras
import keras.backend.tensorflow_backend as K
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'test3_lldel_FClight.h5'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
with K.tf.device('/gpu:0'):
	model = Sequential()

# 32x32x32
	model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

# 16x16x64
	model.add(Conv2D(64, (5, 5), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

# 8x8x128

	model.add(Conv2D(128, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(256, (3, 3), padding='same'))
	model.add(Activation('relu'))

# FC
	model.add(Flatten())
	model.add(Dense(1024, activation='relu'))
	model.add(Dense(10, activation='softmax'))

	opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)


	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())

	x_train=x_train.astype('float32')
	x_test=x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	model.fit(x_train, y_train, epochs = 50, validation_data = (x_test, y_test), batch_size=32)

if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

scores = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])


