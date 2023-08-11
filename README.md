# brainbow color clustering

python reimplementation for brainbow neural tracing:

[1] Duan et al., 2021, [paper](https://www.biorxiv.org/content/10.1101/2020.06.07.138941v1) [code](https://github.com/tuffr5/BrainbowTracing)

[2] Sumbul et al., 2026, [paper](https://proceedings.neurips.cc/paper/2016/hash/7cce53cf90577442771720a370c3c723-Abstract.html)

## (1) denoising
- bm4d for each channel
- call separately as it needs long time and a lot of RAM

## (2) supervoxelize, reimplement from [2]
### (2.1) watershed on sobel
### (2.2) add thresholded foreground
--> which color proximity function?
### (2.3) split supervoxel with different colors
--> not implemented yet
### (2.4) demix/merge supervoxel
--> not implemented yet

## (3) GMM color cluster, from here [1]
- which color distance function?
- how to define dimension d for feature X^(Nxd)?
- how to define number of components for GMM?

## (4) bridge skeletons
--> not implemented yet
- should be simple to implement to verify which cluster should be connected
- which distance threshold?

## other todo's
- pipeline script
- gridsearch for hyperparameters on validation set
- submit evaluate instance segmentation
