import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import GenOpt as opt

# CNN Digit Classification on MNIST Dataset
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

epochs = 12


# Hyperparameter Tuning with GenOpt

# Step 1. Define variables
batch_size = opt.Discrete(100, 1000)
act = opt.Categorical(['sigmoid', 'relu'])
flt1 = opt.Discrete(10, 100)
ker1 = opt.Discrete(2, 7)
flt2 = opt.Discrete(10, 100)
ker2 = opt.Discrete(2, 7)
dpt1 = opt.Continuous(0.1, 0.9)
dns1 = opt.Discrete(10, 1000)
dpt2 = opt.Continuous(0.1, 0.9)
op = opt.Categorical(['Adam', 'SGD', 'RMSprop', 'Adadelta'])

varsCNN = [batch_size, act, flt1, ker1, flt2, ker2, dpt1, dns1, dpt2, op]

# Step 2: define evaluation function
def evalCNN(params):
    
    # tuple-unpacking to extract hyperparameter values
    batch_size, act, flt1, ker1, flt2, ker2, dpt1, dns1, dpt2, op = params
    
    model = Sequential()
    model.add(Conv2D(flt1, kernel_size=(ker1, ker1),
                     activation=act,
                     input_shape=input_shape))
    model.add(Conv2D(flt2, (ker2, ker2), activation=act))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dpt1))
    model.add(Flatten())
    model.add(Dense(dns1, activation=act))
    model.add(Dropout(dpt2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=op,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=12,
              verbose=0,
              validation_data=(x_test, y_test))
              
    score = model.evaluate(x_test, y_test, verbose=0)
    
    
    return score[0]

# Step 3. Run optimizer
opt.optimize(evalCNN, varsCNN)