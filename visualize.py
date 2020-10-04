#! /usr/bin/python3
import pickle

dataset_path = "./RML2016.10a_dict.pkl"
pickled_dataset = pickle.load(open(dataset_path,'rb'), encoding="latin1") # Dataset is a dictionary in form of {(<modulation>, SNR)}
print(set([mod[0] for mod in  pickled_dataset.keys()])) # Set of all modulations in the dataset


# There's 1k snapshots per modulation
#Each entry consists of many snapshots, each snapshot is in the form of a 3d array, ASSUMING first dimension is real or imaginary

# We will use SNR=16



# import matplotlib.pyplot as plt
# for i in range(0,1000,1):
#     x = X[i][0]
#     y= X[i][1]
#     fig = plt.figure()
#     plt.scatter(x,y,c='blue',label=i)
#     plt.xlabel("I")
#     plt.ylabel("Q")
#     plt.title("Data representation variance in BPSK SNR 12")
#     plt.legend()
#     plt.show()
#     fig.savefig(path+"/IQ trained images/snr/%d.png"  % i)