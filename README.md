# Reduced-3DGS: Memory Footprint Reduction for 3D Gaussian Splatting (Python Package Version)

This repository contains the **refactored Python code for [Reduced-3DGS](https://github.com/graphdeco-inria/reduced-3dgs)**. It is forked from commit [13e7393af8ecd83d69197dec7e4c891b333a7c1c](https://github.com/graphdeco-inria/reduced-3dgs/tree/13e7393af8ecd83d69197dec7e4c891b333a7c1c). The original code has been **refactored to follow the standard Python package structure**, while **maintaining the same algorithms as the original version**.

## Features

* [x] Code organized as a standard Python package
* [x] Pruning
* [x] SH Culling
* [x] Vector quantization by K-Means

## Prerequisites

* [Pytorch](https://pytorch.org/) (v2.4 or higher recommended)
* [CUDA Toolkit](https://developer.nvidia.com/cuda-12-4-0-download-archive) (12.4 recommended, should match with PyTorch version)

## Install (PyPI)

```sh
pip install --upgrade reduced-3dgs
```

## Install (Build from source)

```sh
pip install --upgrade git+https://github.com/yindaheng98/reduced-3dgs.git@main
```
If you have trouble with [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting), you can install it from source:
```sh
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
```

## Install (Development)

Install [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting).
You can download the wheel from [PyPI](https://pypi.org/project/gaussian-splatting/):
```shell
pip install --upgrade gaussian-splatting
```
Alternatively, install the latest version from the source:
```sh
pip install --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
```

```shell
git clone --recursive https://github.com/yindaheng98/reduced-3dgs
cd reduced-3dgs
pip install tqdm plyfile scikit-learn numpy
pip install --target . --upgrade --no-deps .
```

(Optional) If you prefer not to install `gaussian-splatting` in your environment, you can install it in your `reduced-3dgs` directory:
```sh
pip install --target . --no-deps --upgrade git+https://github.com/yindaheng98/gaussian-splatting.git@master
```

## Quick Start

1. Download the dataset (T&T+DB COLMAP dataset, size 650MB):

```shell
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip -P ./data
unzip data/tandt_db.zip -d data/
```

2. Train 3DGS with densification, pruning, and SH culling (same as original 3DGS)
```shell
python -m reduced_3dgs.train -s data/truck -d output/truck -i 30000 --mode densify-prune-shculling
```

3. Render 3DGS
```shell
python -m gaussian_splatting.render -s data/truck -d output/truck -i 30000 --mode densify
```

> 💡 Note: This repository does not include code for creating datasets.
> If you wish to create your own dataset, please refer to [InstantSplat](https://github.com/yindaheng98/InstantSplat) or use [convert.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/convert.py).

> 💡 See [.vscode/launch.json](.vscode/launch.json) for more examples. Refer to [reduced_3dgs.train](gaussian_splatting/train.py) and [gaussian_splatting.render](gaussian_splatting/render.py) for full options.

## API Usage

This project heavily depends on [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting) and only provides some enhanced Trainers and Gaussian models. Therefore, before starting, please refer to [`gaussian-splatting`](https://github.com/yindaheng98/gaussian-splatting) to understand the key concepts about Gaussian models, Dataset, Trainers, and how to use them.

### Pruning

`BasePruningTrainer` prunes the trainer at specified training steps.
```python
from reduced_3dgs.pruning import BasePruningTrainer
trainer = BasePruningTrainer(
    gaussians,
    scene_extent=dataset.scene_extent(),
    dataset=dataset,
    prune_from_iter=1000,
    prune_until_iter=15000,
    prune_interval=100,
    ... # see reduced_3dgs/pruning/trainer.py for full options
)
```

`BasePrunerInDensifyTrainer` integrates pruning with densification.
```python
from reduced_3dgs.pruning import BasePrunerInDensifyTrainer
trainer = BasePrunerInDensifyTrainer(
    gaussians,
    scene_extent=dataset.scene_extent(),
    dataset=dataset,
    mercy_from_iter=3000,
    mercy_until_iter=20000,
    mercy_interval=100,
    densify_from_iter=500,
    densify_until_iter=15000,
    densify_interval=100,
    densify_grad_threshold=0.0002,
    densify_opacity_threshold=0.005,
    prune_from_iter=1000,
    prune_until_iter=15000,
    prune_interval=100,
    prune_screensize_threshold=20,
    ... # see reduced_3dgs/pruning/trainer.py for full options
)
```

### SH Culling

`VariableSHGaussianModel` is the 3DGS model that assigns each 3D Gaussian a different SH degree.

```python
from reduced_3dgs.shculling import VariableSHGaussianModel
gaussians = VariableSHGaussianModel(sh_degree).to(device)
```

`BaseSHCullingTrainer` culls the SH degree of each 3D Gaussian at specified training steps.
```python
from reduced_3dgs.shculling import BaseSHCullingTrainer
trainer = BaseSHCullingTrainer(
    gaussians,
    scene_extent=dataset.scene_extent(),
    dataset=dataset,
    cull_at_steps=[15000],
    ... # see reduced_3dgs/shculling/trainer.py for full options
)
```

### Quantization

`VectorQuantizer` is the basic quantization operator:
```python
gaussians.load_ply("output/truck")
from reduced_3dgs.quantization import VectorQuantizer
quantizer = VectorQuantizer(gaussians, num_clusters=256)
quantizer.save_quantized("output/truck-quantized")
quantizer.load_quantized("output/truck-quantized")
```

`BaseVectorQuantizeTrainer` quantizes the model at specified training steps.
```python
from reduced_3dgs.shculling import BaseSHCullingTrainer
trainer = BaseVectorQuantizeTrainer(
    gaussians,
    spatial_lr_scale=dataset.scene_extent(),
    dataset=dataset,
    num_clusters=256,
    quantizate_from_iter=5000,
    quantizate_until_iter=30000,
    quantizate_interval=1000,
    ... # see reduced_3dgs/shculling/trainer.py for full options
)
```

`VectorQuantizeTrainerWrapper` is a wrapper that integrates the quantization step into any Trainer:
```python
trainer = VectorQuantizeTrainerWrapper(
    trainer,

    num_clusters=num_clusters,
    num_clusters_rotation_re=num_clusters_rotation_re,
    num_clusters_rotation_im=num_clusters_rotation_im,
    num_clusters_opacity=num_clusters_opacity,
    num_clusters_scaling=num_clusters_scaling,
    num_clusters_features_dc=num_clusters_features_dc,
    num_clusters_features_rest=num_clusters_features_rest,

    quantizate_from_iter=quantizate_from_iter,
    quantizate_until_iter=quantizate_until_iter,
    quantizate_interval=quantizate_interval,
)
if load_quantized:
    trainer.quantizer.load_quantized(load_quantized)
# see reduced_3dgs/train.py
```

#### Quantized PLY Format

> 💡 See [reduced_3dgs/quantization/quantizer.py](reduced_3dgs/quantization/quantizer.py) for the code to save and load quantized PLY files.

The `save_quantized` function will produce a point cloud stored in a `.ply` format.

Previously, the layout of this file was one row per primitive, containing a series of parameters in `vertex` elements, namely
* 3 floats for position (`x`,`y`,`z`)
* 3 floats for normal (`nx`,`ny`,`nz`)
* 1 uint for the real part of the rotation quaternion (`rot_re`)
* 1 uint for the imaginary part of the rotation quaternion (`rot_im`)
* 1 uint for opacity (`opacity`)
* 3 uint for scaling (`scale`)
* 1 uint for DC color (`f_dc`)
* 3 uint for SH coefficients (`f_rest_0`, `f_rest_1`, `f_rest_2`)

The codebook quantization introduces some additional changes. For different parameters, you can set different lengths of the codebook. Each attribute's codebook will be stored in different elements. The codebooks are ordered as follows:
* `codebook_rot_re` element contains 1 float for the real part of the rotation quaternion (`rot_re`)
* `codebook_rot_im` element contains 3 floats for the 3 imaginary parts of the rotation quaternion (`rot_im_0`, `rot_im_1`, `rot_im_2`)
* `codebook_opacity` element contains 1 float for the opacity (`opacity`)
* `codebook_scaling` element contains 3 floats for the 3 parameters of scale (`scaling_0`, `scaling_1`, `scaling_2`)
* `codebook_f_dc` element contains 3 floats for the 3 DC color parameters (`f_dc_0`, `f_dc_1`, `f_dc_2`)
* 3 elements `codebook_f_rest_<SH degree>` contains floats for SH coefficients of 3 SH degrees (`f_rest_<SH degree>_<SH coefficients at this degree>`).
SH degree 1 has 3 coefficients `f_rest_0_<0,1,2>` in `codebook_f_rest_0`, 
SH degree 2 has 5 coefficients `f_rest_1_<0,1,2,3,4>` in `codebook_f_rest_1`, 
SH degree 3 has 7 coefficients `f_rest_2_<0,1,2,3,4,5,6>` in `codebook_f_rest_2`.

# Reducing the Memory Footprint of 3D Gaussian Splatting
Panagiotis Papantonakis Georgios Kopanas, Bernhard Kerbl, Alexandre Lanvin, George Drettakis<br>
| [Webpage](https://repo-sam.inria.fr/fungraph/reduced_3dgs/) | [Full Paper](https://repo-sam.inria.fr/fungraph/reduced_3dgs/reduced_3DGS_i3d.pdf) | [Datasets (TODO)](TODO) | [Video](https://youtu.be/EnKE-d7eMds?si=xWElEPf4JgwOAmbB&t=48) | [Other GRAPHDECO Publications](http://www-sop.inria.fr/reves/publis/gdindex.php) | [FUNGRAPH project page](https://fungraph.inria.fr) | <br>
![Teaser image](assets/teaser.jpg)

This repository contains the code of the paper "Reducing the Memory Footprint of 3D Gaussian Splatting", which can be found [here](https://repo-sam.inria.fr/fungraph/reduced_3dgs/).
We also provide the configurations to train the models mentioned in the paper,
as well as the evaluation script that produces the results.

<a href="https://www.inria.fr/"><img height="100" src="assets/logo_inria.png"> </a>
<a href="https://univ-cotedazur.eu/"><img height="100" src="assets/logo_uca.png"> </a>
<a href="https://team.inria.fr/graphdeco/"> <img style="width:90%; padding-right: 15px;" src="assets/logo_graphdeco.png"></a>

Abstract: *3D Gaussian splatting provides excellent visual quality for novel view synthesis, with fast training and real-time rendering; unfortunately, the memory requirements of this method for storing and transmission are
unreasonably high. We first analyze the reasons for this, identifying three main areas where storage can
be reduced: the number of 3D Gaussian primitives used to represent a scene, the number of coefficients for
the spherical harmonics used to represent directional radiance, and the precision required to store Gaussian
primitive attributes. We present a solution to each of these issues. First, we propose an efficient, resolution-aware primitive pruning approach, reducing the primitive count by half. Second, we introduce an adaptive
adjustment method to choose the number of coefficients used to represent directional radiance for each
Gaussian primitive, and finally a codebook-based quantization method, together with a half-float representation
for further memory reduction. Taken together, these three components result in a ×27 reduction in overall size
on disk on the standard datasets we tested, along with a ×1.7 speedup in rendering speed. We demonstrate
our method on standard datasets and show how our solution results in significantly reduced download times
when using the method on a mobile device*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{papantonakisReduced3DGS,
      author       = {Papantonakis, Panagiotis and Kopanas, Georgios and Kerbl, Bernhard and Lanvin, Alexandre and Drettakis, George},
      title        = {Reducing the Memory Footprint of 3D Gaussian Splatting},
      journal      = {Proceedings of the ACM on Computer Graphics and Interactive Techniques},
      number       = {1},
      volume       = {7},
      month        = {May},
      year         = {2024},
      url          = {https://repo-sam.inria.fr/fungraph/reduced_3dgs/}
}</code></pre>
  </div>
</section>

