#!/usr/bin/env python 


import numpy as np
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
import glob
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

def chisq_dist(hist1, hist2):

    dist  = np.sqrt(np.nansum(np.divide(np.subtract(hist1,hist2)**2, np.add(hist1, hist2))))

    return dist

def create_distance_matrix(hists):
    
    dm = []
    
    for hist0 in hists:
        line = []
        for hist1 in hists:
            line.append(chisq_dist(hist0,hist1))
        dm.append(line)

    return np.array(dm)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def test_distance_matrix(distmatrix):

    agg = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='average')

    agg.fit(distmatrix)

    plt.imshow(distmatrix, origin='lower', cmap='magma')
    plt.colorbar()
    plt.savefig('distance_matrix.pdf')
    plt.show()

    plot_dendrogram(agg, truncate_mode="level", p=3)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.ylabel('Euclidean Distance')
    plt.savefig('dendogram.pdf')
    plt.show()

    return 

hists = []
hists_flattened = []


model_names = []


for hist in glob.glob('*/*/*log*npy'):
    print(hist)
    model_names.append(hist.split('/')[0])
    hist_array = np.load(hist)
    hist_array = np.nan_to_num(hist_array, posinf=0, neginf=0)
    hists.append(hist_array)
    hists_flattened.append(hist_array.flatten())

hists = np.array(hists)
hists_flattened = np.array(hists_flattened)
model_names = np.array(model_names)


# hists_flattened = np.where(hists_flattened == -1*np.inf, 0, hists_flattened)
# hists_flattened = np.where(hists_flattened == np.inf, 0, hists_flattened)


dist_matrix = distance_matrix(hists_flattened, hists_flattened)
chdists_matrix = create_distance_matrix(hists_flattened)
print(chdists_matrix)



test_distance_matrix(dist_matrix)
# test_distance_matrix(chdists_matrix)


agg_dist = AgglomerativeClustering(distance_threshold=27, n_clusters=None, affinity='precomputed', linkage='average').fit(dist_matrix)
agg_chi = AgglomerativeClustering(distance_threshold=20, n_clusters=None, affinity='precomputed', linkage='average').fit(chdists_matrix)


labels_dist = agg_dist.labels_
labels_chi = agg_chi.labels_

for i in range(8):
    print(f"cluster {i}")
    indices = np.where(labels_dist == i)[0]
    plots_to_view = hists[indices]
    model_names_to_view = model_names[indices]
    mean_vel_hist = np.nanmean(plots_to_view, axis=0)
    for j,h in enumerate(plots_to_view):
        print(model_names_to_view[int(j)])
        plt.imshow(h, origin='lower', vmin=0, vmax=6)
        plt.colorbar()
        plt.show()
    plt.imshow(mean_vel_hist, origin='lower', vmin=0, vmax=6)
    plt.title('mean')
    plt.colorbar()
    plt.savefig(f'cluster_{i}_mean_hist_plot.pdf')
    plt.show()


# for i in range(8):
#     print(i)
#     indices = np.where(labels_chi == i)[0]
#     print(indices)
#     print(hists.shape)
#     plots_to_view = hists[indices]
#     print(plots_to_view.shape)
#     model_names_to_view = model_names[indices]
#     for j,h in enumerate(plots_to_view):
#         print(model_names_to_view[int(j)])
#         plt.imshow(h, origin='lower')
#         plt.show()


# agg_dist = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='average')

# agg_dist.fit(dist_matrix)

# agg_chi = AgglomerativeClustering(distance_threshold=0, n_clusters=None, affinity='precomputed', linkage='average')

# agg_chi.fit(chisq_dist)

# labels_dist = agg_dist.labels_
# labels_chi = agg_chi.labels_


# for labels in set(labels_)

