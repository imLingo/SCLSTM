import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix


def apply_weight_sharing(model, bits=7):
    r'''Applies weight sharing to the given model'''
    counts = 0
    for module in model[1].children():
        if str(module) in ['Sigmoid()','Tanh()','LogSoftmax()','Softmax()','Dropout(p=0.45)']:
            continue
        print(module)
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        print("mat:",mat)
        min_ = min(mat.data)
        max_ = max(mat.data)
        space = np.linspace(min_, max_, num=2 ** bits)
        print("space:",space,space.shape,len(space))
        kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1, 1), n_init=1
                        , precompute_distances=True, algorithm="full")
        kmeans.fit(mat.data.reshape(-1, 1))
        print("label:",kmeans.labels_,len(kmeans.labels_))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        mat.data = new_weight
        module.weight.data = torch.from_numpy(mat.toarray()).to(dev)
        counts = np.prod(weight.shape)
        k=len(space)
        r=(counts * 32)/(counts * np.log2(k) + k * 32)
        print("r=",r)
        print("Cr=",f'{100-(((counts * 32)-(counts * np.log2(k) + k * 32))/(counts * 32))*100:6.2f}%')
        print("Cr=",f'{((counts * np.log2(k) + k * 32)/(counts * 32))*100:6.2f}')
