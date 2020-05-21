#!/ddn/home4/r2444/anaconda3/bin/python
import pandas as pd
import numpy as np
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image
import copy
from math import sqrt

def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

'''
def generate_patches(image_sample, max_patch_no):
    image_sample = np.reshape(image_sample,(28,28))
    patches = image.extract_patches_2d(image_sample, (4, 4), max_patches=max_patch_no,
                                                                        random_state=37)
    return(patches)
'''

def generate_patches(image_sample):
    image_sample_normalized = np.true_divide(image_sample, 255)
    greyIm = np.reshape(image_sample_normalized, (28,28))

    S=greyIm.shape
    N = 2
    patches = []

    for row in range(S[0]):
        for col in range(S[1]):
            Lx=np.max([0,col-N])
            Ux=np.min([S[1],col+N])
            Ly=np.max([0,row-N])
            Uy=np.min([S[0],row+N])
            patch = greyIm[Ly:Uy,Lx:Ux]
            if len(patch) == 4 and len(patch[0]) == 4:
                patches.append(patch)
    return(patches)


def find_nearest_patch(codebook_patches, sample_patches):
    encoded_sample = []
    for i in range(len(sample_patches)):
        patch_distances = []
        for j in range(len(codebook_patches)):
            sample_patch = sample_patches[i].flatten()
            euclid_dist = euclidean_distance(codebook_patches[j], sample_patch)
            patch_distances.append(euclid_dist)
        #print(patch_distances)
        index_min = np.argmin(patch_distances)
        encoded_sample.append(index_min)
    return(encoded_sample)

data = pd.read_csv('mnist_test_euclidean_reduced.csv', header=None)
features= data.columns[1:]
label = data.columns[0]
data_features = data[features]
data_label = data[label]
data_features_np = data_features.to_numpy()
data_label_np = data_label.to_numpy()
data_features_np = data_features_np/ 255

codebooks = pd.read_csv('mnist_train_euclidean_cluster_centers.csv', header=None)
features = codebooks.columns[0:-1]
label = codebooks.columns[-1]
codebook_patches_pd = codebooks[features]
#data_label = data[label]
codebook_patches_np = codebook_patches_pd.to_numpy()
#data_label_np = data_label.to_numpy()


#codebook_patches = generate_patches(codebooks_vector, 100)

new_dataset = []
for i in range(len(data_features_np)):
    temp_new_sample = []
    sample_patches = generate_patches(data_features_np[i])
    temp_new_sample.append(int(data_label_np[i]))
    temp_new_sample += find_nearest_patch(codebook_patches_np, sample_patches)
    new_dataset.append(temp_new_sample)

new_data_pd = pd.DataFrame(new_dataset)

new_data_pd.to_csv('quantized_mnist_euclidean.csv', encoding='utf-8', index=False, header=None)
