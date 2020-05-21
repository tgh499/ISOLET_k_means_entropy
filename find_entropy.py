import image_slicer
import pandas as pd
import numpy as np
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
import copy


def entropy(signal):
    '''
    function returns entropy of a signal
    signal must be a 1-D numpy array
    '''
    lensig=signal.size
    symset=list(set(signal))
    #numsym=len(symset)
    propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
    print(propab)
    ent= np.sum([p*np.log2(1.0/p) for p in propab])
    print(ent)
    return ent

data_original = pd.read_csv('quantized_mnist_original.csv', header=None)
features= data_original.columns[1:]
label = data_original.columns[0]
data_original_features = data_original[features]
data_original_label = data_original[label]
data_original_features_np = data_original_features.to_numpy()
data_original_label_np = data_original_label.to_numpy()

data_shuffled = pd.read_csv('quantized_mnist_shuffled.csv', header=None)
features_shuffled = data_shuffled.columns[1:]
label_shuffled = data_shuffled.columns[0]
data_features_shuffled = data_shuffled[features_shuffled]
data_label_shuffled = data_shuffled[label_shuffled]
data_features_np_shuffled = data_features_shuffled.to_numpy()
data_label_np_shuffled = data_label_shuffled.to_numpy()


data_tsne = pd.read_csv('quantized_mnist_js.csv', header=None)
features_tsne = data_tsne.columns[1:]
label_tsne = data_tsne.columns[0]
data_features_tsne = data_tsne[features_tsne]
data_label_tsne = data_tsne[label_tsne]
data_features_np_tsne = data_features_tsne.to_numpy()
data_label_np_tsne = data_label_tsne.to_numpy()

data_euclidean = pd.read_csv('quantized_mnist_euclidean.csv', header=None)
features_euclidean = data_euclidean.columns[1:]
label_euclidean = data_euclidean.columns[0]
data_features_euclidean = data_euclidean[features_euclidean]
data_label_euclidean = data_euclidean[label_euclidean]
data_features_np_euclidean = data_features_euclidean.to_numpy()
data_label_np_euclidean = data_label_euclidean.to_numpy()


result_original = []
result_shuffled = []
result_tsne = []
result = []
for i in range(100):
    temp_result = []
    sample_original_order = data_original_features_np[i]
    sample_tsne_order = data_features_np_tsne[i]
    sample_euclidean_order = data_features_np_euclidean[i]
    sample_shuffled_order = data_features_np_shuffled[i]
    temp_result.append(data_original_label_np[i])

    temp_result.append(entropy(sample_original_order))
    temp_result.append(entropy(sample_tsne_order))
    temp_result.append(entropy(sample_euclidean_order))
    temp_result.append(entropy(sample_shuffled_order))

    result.append(temp_result)


result_pd = pd.DataFrame(result)
result_pd.to_csv('test.csv', encoding='utf-8', index=False, header=None)
