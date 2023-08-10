import os
import argparse
from glob import glob
import numpy as np
import zarr
import h5py
from bm4d import bm4d


def denoise(vol, sigma_psd, clip_max=None):
    print("denoise with sigma psd %f and clip max %i" % (sigma_psd, clip_max))
    # normalize raw
    vol = vol.astype(np.float32)
    if clip_max is not None:
        vol = np.clip(vol, 0, clip_max)
    vol = vol / clip_max
    
    vol_denoised = []
    # heads up: assuming channels first
    for i in range(3):
        print("denoising channel %i" % i)
        vol_channel = bm4d(vol[i], sigma_psd)
        vol_denoised.append(vol_channel)
    vol_denoised = np.stack(vol_denoised)
    vol_denoised = np.clip(vol_denoised, 0, 1)

    return vol_denoised


def call_denoise_per_sample(infn, in_key, out_folder, sigma_psd=0.05, clip_max=1500):
    # create output file and check if it already exists
    outfn = os.path.join(out_folder, os.path.basename(infn).split(".")[0] + ".hdf")
    if os.path.exists(outfn):
        print("%s already exists, skipping..." % outfn)
        return
    
    # open raw data 
    zinf = zarr.open(infn, "r")
    raw = np.array(zinf[in_key])
    print("reading ", os.path.basename(infn), raw.dtype, raw.shape)

    raw_denoised = denoise(raw, sigma_psd, clip_max)
    
    # denoised volume to hdf
    with h5py.File(outfn, "w") as hout:
        hout.create_dataset(
            "volumes/raw_denoised",
            data=raw_denoised.astype(np.float32),
            dtype=np.float32,
            chunks=True,
            compression="gzip"
        )


def main():
    # get input parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default=None,
            help="path to input file")
    parser.add_argument("--in_folder", type=str, default=None,
            help="path to input folder")
    parser.add_argument("--out_folder", type=str, default=".",
            help="path to output folder")
    parser.add_argument("--in_key", type=str, default="volumes/raw",
            help="key to zarr input volume")
    parser.add_argument("--sigma_psd", type=float, default=0.05,
            help="sigma_psd value for bm4d denoising")
    parser.add_argument("--clip_max", type=int, default=1500,
            help="max value to clip data")
            
    args = parser.parse_args()
    # check that either input file or input folder is given
    assert args.in_file is not None or args.in_folder is not None

    #in_folder = "/nrs/saalfeld/maisl/data/flylight/flylight_complete/fold1"
    #out_folder = "/nrs/saalfeld/maisl/flylight_benchmark/brainbow/denoised/psd_0_05"
    #clip_max = 1500
    #sigma_psd = 0.05
    
    if args.in_folder is not None:
        in_files = glob(in_folder + "/*.zarr")
        for infn in in_files:
            call_denoise_per_sample(infn, args.in_key, 
                    args.out_folder, args.sigma_psd, args.clip_max)
    else:
        call_denoise_per_sample(args.in_file, args.in_key,
                args.out_folder, args.sigma_psd, args.clip_max)


if __name__ == "__main__":
    main()

