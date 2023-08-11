import os
from glob import glob
import numpy as np
import h5py
from scipy import ndimage as ndi
from skimage import io
from skimage.filters import sobel, rank
from skimage.segmentation import watershed
from skimage.segmentation import relabel_sequential
#import torch
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import cv2
import argparse
import time
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
    # convert rgb to luv
    image_flattened = np.reshape(image, (1, np.prod(image.shape[0:-1]), 3)) 
    luv_image_flattened = cv2.cvtColor(image_flattened, cv2.COLOR_RGB2Luv)
    luv_image = np.reshape(luv_image_flattened, image.shape)
    
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


def affinity_matrix(sdist_map, cdist_map, thresh_s=3, thresh_c=8):
    print("threshs: ", thresh_s, thresh_c)
    sdist_map_bin = sdist_map < thresh_s
    cdist_map_bin = cdist_map < thresh_c
    
    adjacency_mat = np.logical_or(sdist_map_bin, cdist_map_bin)
    print("adjacency mat: ", np.sum(adjacency_mat))
    adjacency_mat[np.eye(adjacency_mat.shape[0], dtype=bool)] = False
    print("adjacency mat: ", np.sum(adjacency_mat))
    print(np.prod(adjacency_mat.shape))

    affinity_mat = np.exp(-0.002 * np.square(cdist_map)) * adjacency_mat
    
    return affinity_mat


def GMM(p_components, sv_label, sv_image, n_components=50):
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

    return gmm_label, gmm_image


def color_cluster(sv_label, sv_image, thresh_s=3, thresh_c=8, 
        pca_comps=0, gmm_comps=0, debug=False, debug_dir="."):
    
    start = time.time()
    # relabel supervoxels again sequentially
    sv_label, _, _ = relabel_sequential(sv_label)
    # get spatial distance map
    sdist_map = calc_sdist_map(sv_label)
    # get color distance map
    cdist_map = calc_cdist_map(sv_label, sv_image)
    # get affinity map
    affinity_mat = affinity_matrix(sdist_map, cdist_map, 
            thresh_s=thresh_s, thresh_c=thresh_c)
    # perform PCA
    n_comps = pca_comps if pca_comps > 0 else 'mle'
    pca = PCA(n_components=40)
    p_components = pca.fit_transform(affinity_mat)
    print(p_components.shape)
    
    # perform GMM clustering
    # todo: how to get number for n_components?
    gmm_label, gmm_image = GMM(p_components, sv_label, sv_image, n_components=gmm_comps)
    stop = time.time()
    print("color clustering done in %.2f" % (stop - start))
    return gmm_image, gmm_label


def call_color_cluster_per_sample(infn, sv_label_key, sv_image_key, 
        out_folder, thresh_s, thresh_c, pca_comps, gmm_comps, debug=False): 
    # create output file and check if it already exists
    outfn = os.path.join(out_folder, os.path.basename(infn))
    print(outfn)
    #if os.path.exists(outfn):
    #    continue
    
    # open data
    with h5py.File(infn, "r") as inf:
        sv_label = np.array(inf[sv_label_key])
        sv_image = np.array(inf[sv_image_key])
    print(os.path.basename(infn), sv_label.dtype, sv_label.shape)
    print(os.path.basename(infn), sv_image.dtype, sv_image.shape)
    #sv_label = sv_label[100:300, 300:500, 300:500]
    #sv_image = sv_image[100:300, 300:500, 300:500]
    
    gmm_label, gmm_image = color_cluster(sv_label, sv_image, thresh_s, thresh_c,
            pca_comps, gmm_comps, debug, out_folder)
    
    # write results 
    with h5py.File(outfn, 'w') as outf:
        outf.create_dataset(
            "volumes/gmm_label",
            data=gmm_label,
            dtype=np.uint8,
            chunks=True,
            compression="gzip"
        )
        outf.create_dataset(    # not needed?
            "volumes/gmm_image",
            data=gmm_image,
            dtype=np.uint8,
            chunks=True,
            compression="gzip"
        )
    mip = np.max(gmm_image, axis=0)
    io.imsave(os.path.join(out_folder, "mip_gmm_image.tif"), mip)
    mip = np.max(gmm_label, axis=0)
    io.imsave(os.path.join(out_folder, "mip_gmm_label.tif"), mip)


def main():
    # get input parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default=None,
            help="path to input file")
    parser.add_argument("--in_folder", type=str, default=None,
            help="path to input folder")
    parser.add_argument("--out_folder", type=str, default=".",
            help="path to output folder")
    parser.add_argument("--sv_label_key", type=str, 
            default="volumes/supervoxel_label",
            help="key to hdf supervoxel label volume")
    parser.add_argument("--sv_image_key", type=str, 
            default="volumes/supervoxel_image",
            help="key to hdf supervoxel image volume, containing mean color")
    parser.add_argument("--thresh_s", type=float, default=3,
            help="threshold to filter spatial distances")
    parser.add_argument("--thresh_c", type=float, default=8,
            help="threshold to filter color distances")
    parser.add_argument("--pca_comps", type=int, default=0,
            help="number of components for pca, if 0 set automatically")
    parser.add_argument("--gmm_comps", type=int, default=50,
            help="number of components for gmm, if 0 set automatically")
    parser.add_argument("--debug", default=False,
            action="store_true", help="if yes, intermediate outputs are saved")
            
    args = parser.parse_args()
    # check that either input file or input folder is given
    assert args.in_file is not None or args.in_folder is not None

    #in_folder = "/nrs/saalfeld/maisl/flylight_benchmark/brainbow/supervoxel"
    #out_folder = "/nrs/saalfeld/maisl/flylight_benchmark/brainbow/clustered"
    
    if args.in_folder is not None:
        in_files = glob(in_folder + "/*.hdf")
        for infn in in_files:
            call_color_cluster_per_sample(infn, 
                    args.sv_label_key, args.sv_image_key, 
                    args.out_folder, args.thresh_s, args.thresh_c,
                    args.pca_comps, args.gmm_comps, args.debug)
    else:
        call_color_cluster_per_sample(args.in_file, 
                args.sv_label_key, args.sv_image_key,
                args.out_folder, args.thresh_s, args.thresh_c,
                args.pca_comps, args.gmm_comps, args.debug)


if __name__ == "__main__":
    main()

