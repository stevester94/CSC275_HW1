#! /usr/bin/env python3

import pickle
import numpy as np
import torch

class OShea_DS(torch.utils.data.Dataset):
    def __init__(self, path:str="./RML2016.10a_dict_16snr_only.pkl", min_snr:int=None, max_snr:int=None) -> None:
        """
        args:
            domain_configs: {
                "domain_index":int,
                "min_rotation_degrees":float,
                "max_rotation_degrees":float,
                "num_examples_in_domain":int,
            }
        """
        super().__init__()

        self.modulation_mapping = {
            'AM-DSB': 0,
            'QPSK'  : 1,
            'BPSK'  : 2,
            'QAM64' : 3,
            'CPFSK' : 4,
            '8PSK'  : 5,
            'WBFM'  : 6,
            'GFSK'  : 7,
            'AM-SSB': 8,
            'QAM16' : 9,
            'PAM4'  : 10,
        }

        Xd = pickle.load(open(path,'rb'), encoding="latin1")
        self.Xd = Xd
        snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
        data = []  
        lbl = []
        for mod in mods:
            for snr in snrs:

                if (max_snr == None or snr < max_snr) and (min_snr == None or snr > min_snr):
                    for x in Xd[(mod,snr)]:
                        data.append(
                            (x.astype(np.single), self.modulation_to_int(mod), snr)
                        )


        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def modulation_to_int(self, modulation:str):
        return self.modulation_mapping[modulation]

    def get_snrs(self):
        return list(set(map(lambda i: i[2], self.data)))
        


if __name__ == "__main__":
    ds = OShea_DS()

    for x in ds:
        print(x[0].shape)

    print(ds.get_snrs())