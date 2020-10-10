#! /usr/bin/python3
###########
# Block 1 #
###########

# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
# %matplotlib inline
# import os,random
# os.environ["KERAS_BACKEND"] = "theano"
# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["THEANO_FLAGS"]  = "device=gpu%d"%(1)
import numpy as np
# import theano as th
# import theano.tensor as T
# from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import keras

# from keras.regularizers import *
# from keras.optimizers import adam
import matplotlib.pyplot as plt
# import seaborn as sns
import pickle, random, sys


###########
# Block 2 #
###########

# Load the dataset ...
#  You will need to seperately download or generate this file
# Xd = cPickle.load(open("RML2016.10a_dict.dat",'rb'))
Xd = pickle.load(open("./RML2016.10a_dict.pkl",'rb'), encoding="latin1")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

###########
# Block 3 #
###########

# Partition the data
#  into training and test sets of the form we can train/test on 
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5

train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False)

test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]

# Butchered the fuck out of this to make it py3 compliant, yeesh
def to_onehot(yy):
    l = list(yy)
    maximum_encoding = max(l)
    len_input        = len(l)

    yy1 = np.zeros([len_input, maximum_encoding+1]) # Builds a big ol' m by n matrix

    for index,val in enumerate(l):
        yy1[index][val] = 1
    
    print(yy1)

    return yy1

# The lambda gets the index of the modulation in our big list of modulations
# Resulting map is just a big ol' 1d list of ints
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))


###########
# Block 4 #
###########

in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods

###########
# Block 5 #
###########

# Build VT-CNN2 Neural Net model using Keras primitives -- 

#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization


print("Input Shape: ", in_shp)
dr = 0.5 # dropout rate (%)
model = models.Sequential()

#  - Reshape [N,2,128] to [N,1,2,128] on input
# SM: I don't really get why this is necessary, or how this is done on a matrix level, but ok...
# I guess you can think of this shape as a very long narrow image
# But that extra dimension is throwing me off. Oh well
model.add(
    Reshape([1]+in_shp, input_shape=in_shp)
)

# (None, 1, 2, 128)
print("Reshape output shape: ", model.output_shape)



# Pad zeros around everything, I believe this is in support of the convolution stage
model.add(
    # (symmetric_height_pad, symmetric_width_pad)
    ZeroPadding2D((0, 2))
)
#       (height, width, uh_depth?)
# (None, 1,      6,     128)
print("ZeroPad output shape: ", model.output_shape)


print("Convolution2D (First) Input Shape", model.output_shape)
model.add(
# Originally: Convolution2D(256,1,3, border_mode='valid', activation="relu", name="conv1", init='glorot_uniform')
    Convolution2D(
        filters=256, 
        kernel_size=1,
        strides=3, 
        activation="relu", 
        kernel_initializer='glorot_uniform')
)
print("Convolution2D (First) Output Shape", model.output_shape)

# The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. 
# Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))


print("Convolution2D (Second) Input Shape", model.output_shape)
model.add(
    Convolution2D(filters=80, 
        kernel_size=1, # SM WARNING: Originally 2
        strides=3,
        activation="relu",
        kernel_initializer='glorot_uniform')
)

model.add(Dropout(dr))



model.add(Flatten())
print("Flatten Output Shape: ", model.output_shape)

# SM: So the number of units is the output number, input can be anything (as long as one dimensional)
model.add(
    # Originally:     Dense(256,activation='relu',init='he_normal',name="dense1")
    Dense(
        units=256,
        activation='relu',
        kernel_initializer='he_normal' # I ASSUME kernel is what was initialized using he_normal
    )  
)

print("Dense (First) Output Shape: ", model.output_shape)

model.add(Dropout(dr))
model.add(
    # SM: Weird this did not come with an activation
    Dense(
        units=len(classes),
        kernel_initializer='he_normal')
)
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()


###########
# Block 6 #
###########

# Set up some params 
# SM: Originally 100
nb_epoch = 2     # number of epochs to train on
batch_size = 1024  # training batch size

###########
# Block 7 #
###########

# perform training ...
#   - call the main training loop in keras for our network+dataset

filepath = 'convmodrecnets_CNN2_0.5.wts.h5'

history = model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_data=(X_test, Y_test),
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ]
)

# we re-load the best weights once training is finished
model.load_weights(filepath)



###########
# Block 8 #
###########

# Show simple version of performance
# SM: this is "Scalar test loss"
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print(score)

###########
# Block 9 #
###########

# Show loss curves 
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()

plt.savefig("training_performance.png")
plt.figure()

############
# Block 10 #
############
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

############
# Block 11 #
############
# Plot confusion matrix
test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)

plt.savefig("confusion.png")
plt.figure()


############
# Block 12 #
############

# Plot confusion matrix
acc = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    print(test_X_i.shape)
    test_Y_i_hat = model.predict(test_X_i)

    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

    plt.savefig(str(snr))
    plt.figure()

    plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)

############
# Block 13 #
############

# Save results to a pickle file for plotting later
# print(acc)
# fd = open('results_cnn2_d0.5.dat','wb')
# cPickle.dump( ("CNN2", 0.5, acc) , fd )

############
# Block 14 #
############
# Plot accuracy curve
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
plt.savefig("accuracy_curve.png")