#! /usr/bin/python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy

dataset_in_path = "./RML2016.10a_dict_16snr_only.pkl"

# Dataset is a dictionary in form of {(<modulation>, SNR)}
# Note the "latin1" thing is necessary because this was pickled in python2
pickled_dataset = pickle.load(open(dataset_in_path,'rb'), encoding="latin1") 



# There's 1k snapshots per modulation
#Each entry consists of many snapshots, each snapshot is in the form of a 2d array, ASSUMING first dimension is real or imaginary
print(pickled_dataset.keys())


# IQ Plane
fig, graphs = plt.subplots(4,3)
graphs = graphs.flatten()
graphs[-1].axis('off') # We only have 11 modulations
for index, mod in enumerate(list(pickled_dataset.keys())):
    g = graphs[index]
    I = pickled_dataset[mod][0][0]
    Q = pickled_dataset[mod][0][1]
    g.scatter(I,Q,c='blue')
    g.axes.get_xaxis().set_visible(False)
    g.axes.get_yaxis().set_visible(False)
    g.title.set_text("{}".format(mod[0]))
plt.show()

# Time Domain
fig, graphs = plt.subplots(4,3)
graphs = graphs.flatten()
graphs[-1].axis('off') # We only have 11 modulations
for index, mod in enumerate(list(pickled_dataset.keys())):
    g = graphs[index]
    I = pickled_dataset[mod][0][0]
    Q = pickled_dataset[mod][0][1]
    g.plot(I)
    g.plot(Q)
    g.axes.get_xaxis().set_visible(False)
    g.axes.get_yaxis().set_visible(False)
    g.title.set_text("{}".format(mod[0]))
plt.show()

# Frequency Domain
fig, graphs = plt.subplots(4,3)
graphs = graphs.flatten()
graphs[-1].axis('off') # We only have 11 modulations
for index, mod in enumerate(list(pickled_dataset.keys())):
    g = graphs[index]
    I = pickled_dataset[mod][0][0]
    Q = pickled_dataset[mod][0][1]
    complex_in = I + 1j*Q # Zip em together

    fft = scipy.fft(complex_in)

    num_samps = len(complex_in)
    fft_x = np.linspace(0.0, num_samps//2, num_samps//2)
    fft_y = 2/num_samps * np.abs(fft[0:num_samps//2])
    
    g.plot(fft_x, fft_y)
    g.axes.get_xaxis().set_visible(False)
    g.axes.get_yaxis().set_visible(False)
    g.title.set_text("{}".format(mod[0]))
plt.show()