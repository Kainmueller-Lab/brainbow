import os
from glob import glob
import numpy as np
import h5py
from scipy import ndimage as ndi
from skimage import io
from skimage.filters import sobel, rank
from skimage.segmentation import watershed
from skimage.segmentation import relabel_sequential
import torch
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import cv2
import pdb


def calc_sdist_map(label):
    unique_label = np.unique(label[label > 0])
    num_label = len(unique_label)

    # store supervoxels in a list
    supervoxels = [np.transpose(np.nonzero(label == i)).tolist() for i in unique_label]

    # initialize matrix to store the distances
    d_matrix = np.zeros((num_label, num_label), dtype=np.float32)

    # Calculate distance
    for i in range(num_label):
        supervoxel_a = supervoxels[i]
        for j in range(i + 1, num_label):
            supervoxel_b = supervoxels[j]
            closest_distance = distance.cdist(supervoxel_a, supervoxel_b).min()
            d_matrix[i][j] = d_matrix[j][i] = closest_distance

    return d_matrix


def calc_cdist_map(label, image):
    print(np.min(image, axis=(0,1,2)), np.max(image, axis=(0,1,2)), image.dtype)
    image = image.astype(np.float32) / 255.0
    
    print(image.shape)
    print(np.min(image, axis=(0,1,2)), np.max(image, axis=(0,1,2)), image.dtype)
    image_flattened = np.reshape(image, (1, np.prod(image.shape[0:-1]), 3)) 
    luv_image_flattened = cv2.cvtColor(image_flattened, cv2.COLOR_RGB2Luv)
    luv_image = np.reshape(luv_image_flattened, image.shape)
    print(np.min(luv_image, axis=(0,1,2)), np.max(luv_image, axis=(0,1,2)), luv_image.dtype)
    
    print(label.shape)
    print(np.min(label, axis=(0,1,2)), np.max(label, axis=(0,1,2)), label.dtype)


    print("image: ", image.shape, image.dtype)
    
    unique_label = np.unique(label[label > 0])
    num_label = len(unique_label)

    # Store the supervoxels in a list
    # todo: assert that all colors within one supervoxel are the same? 
    # heads up: only taking one color value as they should be the same for each pixel 
    # in the supervoxel
    supervoxels = [luv_image[label == i].tolist()[0] for i in unique_label]

    # initialize matrix to store the distances
    #d_matrix = np.zeros((num_label, num_label), dtype=np.float32)
    # todo: which distance between colors?
    d_matrix = distance.cdist(supervoxels, supervoxels).astype(np.float32)

    return d_matrix


def affinity_matrix(sdist_map, cdist_map, thresh_s=50, thresh_c=0.5):
    sdist_map_bin = sdist_map < thresh_s
    cdist_map_bin = cdist_map < thresh_c
    
    adjacency_mat = np.logical_or(sdist_map_bin, cdist_map_bin) > 0
    print("adjacency mat: ", np.sum(adjacency_mat))
    adjacency_mat[np.eye(adjacency_mat.shape[0], dtype=bool)] = False
    print("adjacency mat: ", np.sum(adjacency_mat))

    affinity_mat = np.exp(-0.002 * np.square(cdist_map)) * adjacency_mat
    
    return affinity_mat


def GMM(p_components, sv_label, sv_image, n_components=50):
    pdb.set_trace()
    gmm = GaussianMixture(n_components=n_components)

    cluster_label = gmm.fit_predict(p_components)
    print(cluster_label)
    len(cluster_label)
    cluster_label = cluster_label + 1
    
    # create gmm label
    gmm_label = np.zeros_like(sv_label)
    sv_label_unique = np.unique(sv_label[sv_label > 0])
    for g, s in zip(cluster_label, sv_label_unique):
        gmm_label[sv_label == s] = g

    # create gmm image
    print(sv_image.shape, sv_image.dtype)
    gmm_image = np.zeros_like(sv_image)
    for g in np.unique(cluster_label):
        idxs = np.where(gmm_label == g)
        color = np.mean(sv_image[idxs], axis=0).astype(np.uint8)
        gmm_image[idxs] = color

    return gmm_image, gmm_label


in_folder = "/nrs/saalfeld/maisl/flylight_benchmark/brainbow/supervoxel"
out_folder = "/nrs/saalfeld/maisl/flylight_benchmark/brainbow/clustered"

in_files = glob(in_folder + "/*.hdf")
for in_fn in in_files:
    # create output file and check if it already exists
    out_fn = os.path.join(out_folder, os.path.basename(in_fn))
    print(out_fn)
    #if os.path.exists(out_fn):
    #    continue
    
    # open data
    with h5py.File(in_fn, "r") as inf:
        sv_label = np.array(inf["volumes/supervoxel_label"])
        #sv_label = sv_label[100:300, 300:500, 300:500]
        sv_image = np.array(inf["volumes/supervoxel_raw"])
        #sv_image = sv_image[100:300, 300:500, 300:500]
    print(os.path.basename(in_fn), sv_label.dtype, sv_label.shape)
    #raw = np.moveaxis(raw, 0, -1)
    #print(os.path.basename(in_fn), raw.dtype, raw.shape)
    sv_label = sv_label - 1
    sv_label, _, _ = relabel_sequential(sv_label)

    # get spatial distance map
    sdist_map = calc_sdist_map(sv_label)
    # get color distance map
    cdist_map = calc_cdist_map(sv_label, sv_image)
    # get affinity map
    affinity_mat = affinity_matrix(sdist_map, cdist_map, thresh_s=3, thresh_c=8)
    # perform PCA
    pdb.set_trace()
    pca = PCA(n_components='mle')
    p_components = pca.fit_transform(affinity_mat)
    print(p_components.shape)
    
    # perform GMM clustering
    # todo: how to get number for n_components?
    gmm_image, gmm_label = GMM(p_components, sv_label, sv_image, n_components=20)
    
    # write results 
    with h5py.File(out_fn, 'w') as hout:
        hout.create_dataset(
            "volumes/gmm_image",
            data=gmm_image,
            dtype=np.uint8,
            chunks=True,
            compression="gzip"
        )
        
        hout.create_dataset(
            "volumes/gmm_label",
            data=gmm_label,
            dtype=np.uint8,
            chunks=True,
            compression="gzip"
        )

    mip = np.max(gmm_image, axis=0)
    io.imsave(
            os.path.join(os.path.dirname(out_fn), "mip_image.tif"),
            mip
            )
    
    mip = np.max(gmm_label, axis=0)
    io.imsave(
            os.path.join(os.path.dirname(out_fn), "mip_label.tif"),
            mip
            )

