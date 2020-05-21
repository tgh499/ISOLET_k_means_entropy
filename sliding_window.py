import pandas as pd
import numpy as np

def create_patches_add_label(image_sample, image_sample_label):
    data_patches = [[]]
    greyIm = np.true_divide(image_sample, 255)

    S = greyIm.shape
    N = 4
    patches = []

    for col in range(S[0]):
        Lx = np.max([0, col-N])
        Ux = np.min([S[0], col+N])
        patch = greyIm[Lx:Ux]
        if len(patch) == 8:
            patches.append(patch)

    for patch in patches:
        temp_array = []
        for i in patch:
            temp_array.append(i)
        temp_array.append(image_sample_label)
        data_patches.append(temp_array)
        
    return(data_patches)


def main(filename_prefix):
    input_filename = filename_prefix + ".csv"
    output_filename = filename_prefix + "_patches.csv"

    data = pd.read_csv(input_filename, header=None)
    features = data.columns[1:]
    label = data.columns[0]
    data_features = data[features]
    data_label = data[label]
    data_features_np = data_features.to_numpy()
    data_label_np = data_label.to_numpy()

    data_patches = []

    for i in range(len(data_features_np)):
        data_patches.append(create_patches_add_label(data_features_np[i], 
                                                            data_label_np[i]))

    data_patches_pd = pd.DataFrame(data_patches)
    data_patches_pd.to_csv(output_filename,
                        encoding='utf-8', index=False, header=None)

if __name__ == "__main__":
    main('train_randomized')