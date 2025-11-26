#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

rasterizor_root = "submodules/diff-gaussian-rasterization"
rasterizor_sources = [
    "cuda_rasterizer/rasterizer_impl.cu",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/backward.cu",
    "rasterize_points.cu",

    "reduced_3dgs/redundancy_score.cu",
    "reduced_3dgs/sh_culling.cu",
    "reduced_3dgs/kmeans.cu",
    "reduced_3dgs.cu",

    "ext.cpp"]
simpleknn_root = "submodules/simple-knn"
simpleknn_sources = [
    "spatial.cu",
    "simple_knn.cu",
    "ext.cpp"]
packages = ['reduced_3dgs'] + ["reduced_3dgs." + package for package in find_packages(where="reduced_3dgs")]
rasterizor_packages = {
    'reduced_3dgs.diff_gaussian_rasterization': 'submodules/diff-gaussian-rasterization/diff_gaussian_rasterization',
    'reduced_3dgs.simple_knn': 'submodules/simple-knn/simple_knn',
}

importance_root = "submodules/gaussian-importance"
importance_sources = [
    "cuda_rasterizer/rasterizer_impl.cu",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/backward.cu",
    "rasterize_points.cu",
    "ext.cpp"]
importance_packages = {
    'reduced_3dgs.importance.diff_gaussian_rasterization': 'submodules/gaussian-importance/diff_gaussian_rasterization',
}

cxx_compiler_flags = []
nvcc_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")
    nvcc_compiler_flags.append("-allow-unsupported-compiler")

setup(
    name="reduced_3dgs",
    version='1.12.2',
    author='yindaheng98',
    author_email='yindaheng98@gmail.com',
    url='https://github.com/yindaheng98/reduced-3dgs',
    description=u'Refactored code for the paper "Reducing the Memory Footprint of 3D Gaussian Splatting"',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=packages + list(rasterizor_packages.keys()) + list(importance_packages.keys()),
    package_dir={
        'reduced_3dgs': 'reduced_3dgs',
        **rasterizor_packages,
        **importance_packages,
    },
    ext_modules=[
        CUDAExtension(
            name="reduced_3dgs.diff_gaussian_rasterization._C",
            sources=[os.path.join(rasterizor_root, source) for source in rasterizor_sources],
            extra_compile_args={"nvcc": nvcc_compiler_flags + ["-I" + os.path.join(os.path.abspath(rasterizor_root), "third_party/glm/")]}
        ),
        CUDAExtension(
            name="reduced_3dgs.importance.diff_gaussian_rasterization._C",
            sources=[os.path.join(importance_root, source) for source in importance_sources],
            extra_compile_args={"nvcc": nvcc_compiler_flags + ["-I" + os.path.join(os.path.abspath(importance_root), "third_party/glm/")]}
        ),
        CUDAExtension(
            name="reduced_3dgs.simple_knn._C",
            sources=[os.path.join(simpleknn_root, source) for source in simpleknn_sources],
            extra_compile_args={"nvcc": nvcc_compiler_flags, "cxx": cxx_compiler_flags}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'gaussian-splatting',
        'scikit-learn',
    ]
)
