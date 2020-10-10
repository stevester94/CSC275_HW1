#! /usr/bin/python3
import pickle

dataset_in_path = "./RML2016.10a_dict.pkl"
filtered_dataset_out_path = "./RML2016.10a_dict_16snr_only.pkl"

pickled_dataset = pickle.load(open(dataset_in_path,'rb'), encoding="latin1") 

snr_16 = {}

for mod in pickled_dataset.keys():
    if mod[1] == 16:
        snr_16[mod] = pickled_dataset[mod]

pickle.dump(snr_16, open(filtered_dataset_out_path, "wb"))