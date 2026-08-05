import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
site_packages_dir = os.path.dirname(os.path.dirname(torch.__file__))
cuda_include_dirs = []
cuda_home = os.environ.get("CUDA_HOME")
if cuda_home:
    for include_dir in (
        os.path.join(cuda_home, "include"),
        os.path.join(cuda_home, "targets", "x86_64-linux", "include"),
    ):
        if os.path.isdir(include_dir):
            cuda_include_dirs.append(include_dir)
cuda_library_dirs = [torch_lib_dir]
cuda_runtime_lib_dir = os.path.join(site_packages_dir, "nvidia", "cuda_runtime", "lib")
if os.path.isdir(cuda_runtime_lib_dir):
    cudart_link = os.path.join(cuda_runtime_lib_dir, "libcudart.so")
    if not os.path.exists(cudart_link):
        for filename in sorted(os.listdir(cuda_runtime_lib_dir)):
            if filename.startswith("libcudart.so."):
                os.symlink(filename, cudart_link)
                break
    cuda_library_dirs.append(cuda_runtime_lib_dir)

extra_compile_args = {
    "cxx": [
        "-g", 
        "-O3", 
        "-fopenmp", 
        "-lgomp", 
        "-std=c++17",
        "-DENABLE_BF16"
    ],
    "nvcc": [
        "-O3", 
        "-std=c++17",
        "-DENABLE_BF16",  # TODO
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=8"
    ],
}

setup(
    name="patternkv_gemv",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="patternkv_gemv",
            sources=[
                "csrc/pybind.cpp", 
                "csrc/gemv_cuda.cu"
            ],
            include_dirs=cuda_include_dirs,
            extra_compile_args=extra_compile_args,
            library_dirs=cuda_library_dirs,
            extra_link_args=[f"-Wl,-rpath,{path}" for path in cuda_library_dirs],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
