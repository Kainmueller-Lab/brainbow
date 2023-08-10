import os
from glob import glob
import argparse
import numpy as np
import h5py
from scipy import ndimage, spatial
from skimage import io, filters, measure, segmentation, color
import pdb


# script to create supervoxels
# reimplementation from Sumbul et al., 2016
# https://proceedings.neurips.cc/paper/2016/hash/7cce53cf90577442771720a370c3c723-Abstract.html
# Section 2.2 Dimensionality reduction
# (1) watershed on gradient image --> assign boundaries to 
#     neighboring basins based on color proximity (?)
# (2) topology-preserving warping of thresholded image B=W(I_theta, F)
# (3) dividing non-homogeneous supervoxels
# (4) demixing overlapping neurons
def watershed(image, thresh=0.05, debug=False):
    # (1) create supervoxel with watershed
    # use sobel filter to get boundaries
    gradient = filters.sobel(image, axis=(0, 1, 2))
    gradient = np.max(gradient, axis=-1)
    markers = gradient < thresh
    markers = ndimage.label(markers)[0]
    sv_label = segmentation.watershed(
            gradient, markers, connectivity=np.ones((3,3,3)))
    # set 1 to 0 background, todo: should be better largest component
    sv_label = sv_label - 1
    
    # save intermediate outputs for debugging
    if debug:
        mip = np.max(gradient, axis=0)
        io.imsave("mip_gradient_debug.tif", mip)
        mip = np.max(sv_label, axis=0)
        io.imsave("mip_sv_debug.tif", mip)
    
    # save mean rgb color for each supervoxel
    sv_image = np.zeros_like(image)
    for i in np.unique(sv_label):
        idx = np.where(sv_label == i)
        sv_image[idx] = np.mean(image[idx], axis=0)
        
    sv_label = sv_label.astype(np.int32)
    sv_image = sv_image.astype(np.uint8)

    return sv_label, sv_image


def add_ridges(image, sv_label, sv_image, thresh=0.2, debug=False):
    # (2) add one-pixel-thin processes which got lost in watershed
    # todo: how to threshold image? use of vesselness filter?
    # convert rgb [0..255] to grey [0..1]
    image_grey = color.rgb2gray(image)
    fg_mask = image_grey > thresh
    sv_fg_mask = sv_label > 0
    add_fg = np.logical_and(fg_mask, np.logical_not(sv_fg_mask))
    if np.sum(add_fg) <= 0:
        print("no fg pixel to add, return")
        return sv_label, sv_image
    else:
        print("adding %i pixel to supervoxel" % np.sum(add_fg))
    # convert images to range [0..1]
    image01 = image / 255.0
    sv_image01 = sv_image / 255.0
    
    # save intermediate outputs for debugging
    if debug:
        mip = np.max((image_grey * 255).astype(np.uint8), axis=0)
        io.imsave("mip_grey_debug.png", mip)
        mip = np.max((fg_mask * 255).astype(np.uint8), axis=0)
        io.imsave("mip_grey_debug.png", mip)

    # create kd tree for existing supervoxel
    sv_coords = np.nonzero(sv_fg_mask)
    sv_kdtree = spatial.KDTree(np.transpose(sv_coords))
    # connected component, todo: add min size thresh ?
    cc = measure.label(add_fg)
    cc_ids = np.unique(cc[cc > 0])
    for cc_id in cc_ids:
        cc_coords = np.nonzero(cc == cc_id)
        # get neighboring supervoxel
        # todo: how define distance thresh here?
        n_ids = sv_kdtree.query_ball_point(np.transpose(cc_coords), 2)
        # skip cc if no neighboring supervoxel
        if len(n_ids[0]) == 0:
            continue
        pdb.set_trace()
        n_ids = np.unique(np.concatenate(n_ids))
        n_coords_list = np.transpose(sv_coords)[n_ids]
        n_coords = tuple(np.transpose(n_coords_list))
        # check color proximity, todo: not sure how it is described in paper?
        cc_color = np.mean(image01[cc_coords], axis=0)
        diff = np.max(np.abs(sv_image01[n_coords] - cc_color), axis=1)
        if np.min(diff) < 0.5:
            # add connected component to supervoxel
            sv_id = sv_label[tuple(n_coords_list[np.argmin(diff)])]
            sv_label[cc_coords] = sv_id
            mask = sv_label == sv_id
            sv_image01[mask] = np.mean(image01[mask], axis=0)
        else:
            # label connected component as new supervoxel
            # could update kdtree here, but maybe not necessary
            sv_label[cc_coords] = np.max(sv_label) + 1
            sv_image01[cc_coords] = cc_color

        # convert sv_image back to range [0..255]
        sv_image = (sv_image * 255).astype(np.uint8)

    return sv_label, sv_image


def supervoxelize(image, seed_thresh=0.05, fg_thresh=0.2, debug=False):
    
    sv_label, sv_image = watershed(image, seed_thresh, debug)
    sv_label, sv_image = add_ridges(image, sv_label, sv_image, fg_thresh, debug)

    return sv_label, sv_image


def call_supervoxelize_per_sample(infn, in_key, out_folder, 
        seed_thresh=0.05, fg_thresh=0.2, debug=False):
    # create output file and check if it already exists
    outfn = os.path.join(out_folder, os.path.basename(infn))
    print(outfn)
    #if os.path.exists(out_fn):
    #    continue
    
    # open data
    with h5py.File(infn, "r") as inf:
        raw = np.array(inf[in_key])
        raw = raw[:, 100:390, 500:680, 300:550]
    # change from channels first to channels last
    raw = np.moveaxis(raw, 0, -1)

    # create supervoxel
    sv_label, sv_image = supervoxelize(raw, seed_thresh=0.05, fg_thresh=0.2)
    
    # save supervoxel in hdf
    with h5py.File(outfn, 'w') as outf:
        outf.create_dataset(
                "volumes/supervoxel_image",
                data=sv_image.astype(np.uint8),
                dtype=np.uint8,
                chunks=True,
                compression="gzip"
        )
        outf.create_dataset(
            "volumes/supervoxel_label",
            data=supervoxel_label,
            dtype=supervoxel_label.dtype,
            chunks=True,
            compression="gzip"
        )
    # save mips if debugging is true
    if debug:
        out_base = os.path.dirname(outfn)
        mip = np.max(sv_image, axis=0)
        io.imsave(os.path.join(out_base, "mip_sv_image.tif"), mip)
        mip = np.max(sv_label, axis=0)
        io.imsave(os.path.join(out_base, "mip_label.tif"), mip)


def main():
    in_folder = "/nrs/saalfeld/maisl/flylight_benchmark/brainbow/denoised/psd_0_05"
    out_folder = "/nrs/saalfeld/maisl/flylight_benchmark/brainbow/supervoxel"
    in_key = "volumes/raw_denoised_uint8"
    in_files = glob(in_folder + "/*.hdf")
    seed_thresh = 0.05
    fg_thresh = 0.2
    debug = False

    for infn in in_files:
        call_supervoxelize_per_sample(infn, in_key, out_folder, 
                seed_thresh, fg_thresh, debug)
        
if __name__ == "__main__":
    main()

