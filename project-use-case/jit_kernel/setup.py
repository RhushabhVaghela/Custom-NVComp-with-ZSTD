#!/usr/bin/env python3
"""
setup.py (FINAL - nvCOMP 5.0 Manager API)

Builds JIT decompression backend with nvCOMP 5.0 C++ Manager API.
Correct return types: shared_ptr<nvcompManagerBase>
"""

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import os

def get_gpu_arch():
    """Detect GPU architecture."""
    if not torch.cuda.is_available():
        return ["sm_60", "sm_70"]
    
    try:
        gpu_arch = []
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            major, minor = capability[0], capability[1]
            
            if major == 8 and minor == 9:
                arch = "sm_89"
            elif major >= 12:
                arch = "sm_89"
            else:
                arch = f"sm_{major}{minor}"
            
            if arch not in gpu_arch:
                gpu_arch.append(arch)
        
        safe_archs = ["sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89"]
        for arch in safe_archs:
            if arch not in gpu_arch:
                gpu_arch.append(arch)
        
        valid_archs = []
        for arch in gpu_arch:
            arch_num = int(arch[3:])
            if 50 <= arch_num <= 90:
                valid_archs.append(arch)
        
        return valid_archs[:6]
    except Exception as e:
        print(f"⚠️ Architecture detection failed: {e}")
        return ["sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_89"]

def find_nvcomp_paths():
    """Find nvCOMP 5.0 system installation paths."""
    include_paths = []
    lib_paths = []
    
    print("\n🔍 Searching for nvCOMP (prioritizing system install)...")
    
    system_check_paths = [
        "/usr/include/nvcomp_12",
        "/usr/include/nvcomp",
        "/opt/nvcomp",
    ]
    
    for check_path in system_check_paths:
        if os.path.exists(check_path):
            print(f"✅ Found system install: {check_path}")
            if check_path not in include_paths:
                include_paths.append(check_path)
            break
    
    lib_check_paths = [
        "/usr/lib/x86_64-linux-gnu/nvcomp/12",
        "/usr/lib/x86_64-linux-gnu/nvcomp",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib",
        "/opt/nvcomp/lib",
    ]
    
    for check_path in lib_check_paths:
        if os.path.exists(check_path):
            if check_path not in lib_paths:
                lib_paths.append(check_path)
    
    if include_paths:
        print(f"✅ Using combined system includes: {include_paths}")
        print(f"✅ Using system libs: {lib_paths}")
        return include_paths, lib_paths
    
    print("⚠️ System path not found. Falling back to Conda/Pip paths...")
    
    conda_env = os.environ.get('CONDA_PREFIX')
    if not conda_env:
        print("❌ ERROR: CONDA_PREFIX environment variable not set.")
        return [], []
    
    pip_include = os.path.join(conda_env, "lib/python3.10/site-packages/nvidia/nvcomp/include")
    conda_include = os.path.join(conda_env, "include")
    
    # We MUST prioritize the main conda env include, as the pip package's
    # headers are often structured incorrectly (flat) and have broken internal includes.
    if os.path.exists(conda_include):
        print(f"✅ Found conda env include: {conda_include}")
        include_paths.append(conda_include)
    
    elif os.path.exists(pip_include):
        print(f"✅ Found pip package include (fallback): {pip_include}")
        include_paths.append(pip_include)
    
    if not include_paths:
        print("❌ FATAL: Could not find any nvCOMP include paths.")
    
    return include_paths, lib_paths

def find_pytorch_paths():
    """Auto-discover PyTorch library paths."""
    pytorch_lib_paths = []
    
    print("\n🔍 Searching for PyTorch libraries...")
    
    if os.environ.get('PYTORCH_LIB'):
        pytorch_lib = os.environ['PYTORCH_LIB']
        if os.path.exists(pytorch_lib):
            print(f"✅ Found PYTORCH_LIB: {pytorch_lib}")
            pytorch_lib_paths.append(pytorch_lib)
            return pytorch_lib_paths
    
    try:
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib_path):
            print(f"✅ Found PyTorch lib from package: {torch_lib_path}")
            pytorch_lib_paths.append(torch_lib_path)
            return pytorch_lib_paths
    except:
        pass
    
    conda_env = os.environ.get('CONDA_PREFIX')
    if conda_env:
        conda_pytorch_lib = os.path.join(conda_env, 'lib')
        if os.path.exists(conda_pytorch_lib):
            print(f"✅ Found conda PyTorch lib: {conda_pytorch_lib}")
            pytorch_lib_paths.append(conda_pytorch_lib)
            return pytorch_lib_paths
    
    print("⚠️ No PyTorch lib found, using system default")
    return pytorch_lib_paths

# === AUTO-DISCOVERY ===

cuda_available = torch.cuda.is_available()
gpu_architectures = get_gpu_arch() if cuda_available else []
nvcomp_includes, nvcomp_libs = find_nvcomp_paths()
pytorch_libs = find_pytorch_paths()

print("\n🔥 JIT Decompress CUDA Extension Setup (WITH AUTO-DISCOVERY)")
print("=" * 60)
print(f"CUDA Available: {cuda_available}")
print(f"GPU Architectures: {gpu_architectures}")
print(f"nvCOMP Include Paths: {nvcomp_includes}")
print(f"nvCOMP Library Paths: {nvcomp_libs}")
print(f"PyTorch Library Paths: {pytorch_libs}")
print(f"PyTorch Version: {torch.__version__}")

# === BUILD CONFIGURATION ===

extensions = []

if cuda_available and nvcomp_includes:
    
    # CUDA flags
    cuda_flags = [
        '-O3', '--use_fast_math', '--restrict', '--expt-relaxed-constexpr',
        '-Xptxas=-v', '--extended-lambda',
    ]
    
    # Add GPU architectures
    for arch in gpu_architectures:
        arch_num = arch[3:]
        cuda_flags.extend(['-gencode', f'arch=compute_{arch_num},code=sm_{arch_num}'])
    
    # Add latest arch as compute capability
    if gpu_architectures:
        latest = max(gpu_architectures, key=lambda x: int(x[3:]))
        latest_num = latest[3:]
        cuda_flags.extend(['-gencode', f'arch=compute_{latest_num},code=compute_{latest_num}'])
    
    # Include directories
    include_dirs = []

    # nvCOMP includes (CRITICAL: Must come FIRST to override system paths)
    for nvcomp_path in nvcomp_includes:
        if nvcomp_path not in include_dirs:
            include_dirs.append(nvcomp_path)
    
    # PyTorch includes
    try:
        torch_include = os.path.join(os.path.dirname(torch.__file__), 'include')
        if os.path.exists(torch_include):
            include_dirs.append(torch_include)
    except:
        pass
    
    # CUDA includes
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    cuda_include = os.path.join(cuda_home, 'include')
    if os.path.exists(cuda_include):
        include_dirs.append(cuda_include)
    
    # nvCOMP includes (CRITICAL: Must come before system includes)
    for nvcomp_path in nvcomp_includes:
        if nvcomp_path not in include_dirs:
            include_dirs.append(nvcomp_path)
    
    # Library directories
    library_dirs = []
    for lib_path in nvcomp_libs:
        if lib_path not in library_dirs:
            library_dirs.append(lib_path)
    
    for pytorch_lib in pytorch_libs:
        if pytorch_lib not in library_dirs:
            library_dirs.append(pytorch_lib)
    
    # Link arguments
    extra_link_args = ['-lnvcomp', '-lcudart']
    
    for pytorch_lib in pytorch_libs:
        extra_link_args.append(f'-Wl,-rpath,{pytorch_lib}')
    
    for nvcomp_lib in nvcomp_libs:
        extra_link_args.append(f'-Wl,-rpath,{nvcomp_lib}')
    
    # Build extension
    cuda_extension = CUDAExtension(
        name='jit_kernel_cuda',
        sources=[
            'jit_kernel_pybind.cpp',
            'jit_kernel.cu'
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=['nvcomp'],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++17', '-fPIC'],
            'nvcc': cuda_flags
        },
        extra_link_args=extra_link_args
    )
    
    extensions.append(cuda_extension)
    print("\n✅ CUDA + nvCOMP extension configured successfully")
else:
    print("\n❌ Cannot build extension")
    if not cuda_available:
        print(" Reason: CUDA not available")
    if not nvcomp_includes:
        print(" Reason: nvCOMP headers not found")

# === SETUP ===

setup(
    name='jit-decompress',
    version='2.4.0',
    description='🔥 JIT Decompression with nvCOMP 5.0 Manager API (FINAL)',
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)},
    zip_safe=False,
)
