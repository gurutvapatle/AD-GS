<h1 align="center">AD-GS: Alternating Densification for Sparse-Input 3D Gaussian Splatting</h1>

<p align="center">
  <a href="https://gurutvapatle.github.io/">Gurutva Patle</a><sup>1</sup>, 
  <a href="https://nilzilla03.github.io">Nilay Girgaonkar</a><sup>1</sup>, 
  <a href="https://nagabhushansn95.github.io/">Nagabhushan Somraj</a><sup>1</sup>, 
  <a href="https://ece.iisc.ac.in/~rajivs/#/">Rajiv Soundararajan</a><sup>1</sup>
</p>

<p align="center"><sup>1</sup>Indian Institute of Science (IISc), Bengaluru</p>

<p align="center"><strong>Contact</strong>: Gurutva Patle via email (gurutvac[at]iisc[dot]ac[dot]in).</p>

<p align="center">
  <strong>✨ACM SIGGRAPH Asia 2025✨</strong>
</p>

<p align="center">
  <a href='https://arxiv.org/abs/2509.11003'>
  <img src='https://img.shields.io/badge/Arxiv-2508.13139-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a> 
  <a href='https://arxiv.org/abs/2509.11003'>
  <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a> 
  <a href='https://github.com/gurutvapatle/AD-GS'>
  <img src='https://img.shields.io/badge/GitHub-Code-black?style=flat&logo=github&logoColor=white'>
</a> 
  <a href='https://gurutvapatle.github.io/publications/2025/ADGS.html'>
  <img src='https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323'>
</a> 
</p>



[//]: # (<p align="center">)

[//]: # (  <video width="800" controls>)

[//]: # (    <source src="./assets/cross-topo-retarget.mp4" type="video/mp4">)

[//]: # (    Your browser does not support the video tag.)

[//]: # (  </video>)

[//]: # (</p>)

This is the official repository for our paper **AD-GS: Alternating Densification for Sparse-Input 3D Gaussian Splatting**.


## Abstract

3D Gaussian Splatting (3DGS) has shown impressive results in real-time novel view synthesis. However, it often struggles under sparse-view settings, producing undesirable artifacts such as floaters, inaccurate geometry, and overfitting due to limited observations.

We propose AD-GS, a novel alternating densification framework that interleaves high and low densification phases. During high densification, the model densifies aggressively, followed by photometric loss based training to capture fine-grained scene details. Low densification then primarily involves aggressive opacity pruning of Gaussians followed by regularizing their geometry through pseudo-view consistency and edge-aware depth smoothness.

This alternating approach helps reduce overfitting by carefully controlling model capacity growth while progressively refining the scene representation. Extensive experiments on challenging datasets demonstrate that AD-GS significantly improves rendering quality and geometric consistency compared to existing methods.

## Method Diagram
![method](assets/method.png)

### Tested on 
``````
Ubuntu 20.04.6 LTS, 
NVIDIA-SMI 535.183.01 Driver Version: 535.183.01 CUDA Version: 12.2, 
PyTorch 1.12.1
``````

## Installation

``````
conda env create --file environment.yml
conda activate adgs
``````

``````
pip install submodules/diff-gaussian-rasterization-confidence
pip install submodules/simple-knn
``````

## Sample Run

We provide a sample file for horns scene here in repo. Run below command to run demo.sh .

``````
bash demo.sh
``````

A file "full_logs.json" contains the output.

## Required Data
```
├── /dataset
   ├── mipnerf360
        ├── bicycle
        ├── bonsai
        ├── ...
   ├── nerf_llff_data
        ├── fern
        ├── flower
        ├── ...
   ├── Tanks
        ├── ballroom
        ├── barn
        ├── ...
```

[//]: # (## Preprocessed Dataset)

[//]: # ()
[//]: # (There are the processed datasets used in AD-GS, with which the data preprocessed steps can be skipped:)

[//]: # (1. [Tanks & Temples 3,6,9 views]&#40;will update soon&#41;)

[//]: # (2. [LLFF 3,6,9 views]&#40;will update soon&#41;)

[//]: # (3. [MipNeRF-360 12,24 views]&#40;will update soon&#41;)

## Evaluation

### LLFF

1. Download LLFF from [the official download link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

2. run colmap to obtain initial point clouds with limited viewpoints:
    ```bash
   python tools/colmap_llff.py
   ```

3. Start training and testing:

   ```bash
   # for example 
   # put integer number 0 or 1 or 2 in place of ${gpu_id} in below command
   bash scripts/run_llff.sh ${gpu_id} dataset/nerf_llff_data/horns output/nerf_llff_data/horns
   ```

### MipNeRF-360

1. Download MipNeRF-360 from [the official download link](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip).

2. run colmap to obtain initial point clouds with limited viewpoints:
    ```bash
   python tools/colmap_360.py
   ```
   
3. Start training and testing:

   ```bash
   # for example
   bash scripts/run_360.sh ${gpu_id} dataset/mipnerf360/bicycle output/mipnerf360/bicycle
   ```

### Tanks & Temples

1. Download Tanks & Temples from [download link](https://www.robots.ox.ac.uk/~wenjing/Tanks.zip). 
We refer to [NoPe-NeRF](https://github.com/ActiveVisionLab/nope-nerf/?tab=readme-ov-file) for downloading.

2. run colmap to obtain initial point clouds with limited viewpoints:
    ```bash
   python tools/colmap_llff.py
   ```
   We use the same LLFF colmap_llff.py file. Change the scene names in above file for Tanks & Temples.

3. Start training and testing:

   ```bash
   # for example
   bash scripts/run_tanks.sh ${gpu_id} dataset/tanksntemples/ballroom output/tanksntemples/ballroom
   ```

## Customized Dataset
Similar to Gaussian Splatting, our method can read standard COLMAP format datasets. Please customize your sampling rule in `scenes/dataset_readers.py`, and see how to organize a COLMAP-format dataset from raw RGB images.



## Citation

Consider citing as below if you find this repository helpful to your project:

```
@misc{patle2025adgs,
  author       = {Gurutva Patle and Nilay Girgaonkar and Nagabhushan Somraj and Rajiv Soundararajan},
  title        = {AD‑GS: Alternating Densification for Sparse‑Input 3D Gaussian Splatting},
  journal      = {arXiv preprint arXiv:2509.11003},
  year         = {2025},
  archivePrefix= {arXiv},
  primaryClass = {cs.GR}
}
```

## Acknowledgement

This work was supported in part by the Kotak IISc AI–ML Centre (KIAC).


