# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/build_v32/_deps/zstd-src"
  "/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/build_v32/_deps/zstd-build"
  "/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/build_v32/_deps/zstd-subbuild/zstd-populate-prefix"
  "/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/build_v32/_deps/zstd-subbuild/zstd-populate-prefix/tmp"
  "/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/build_v32/_deps/zstd-subbuild/zstd-populate-prefix/src/zstd-populate-stamp"
  "/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/build_v32/_deps/zstd-subbuild/zstd-populate-prefix/src"
  "/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/build_v32/_deps/zstd-subbuild/zstd-populate-prefix/src/zstd-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/build_v32/_deps/zstd-subbuild/zstd-populate-prefix/src/zstd-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD/build_v32/_deps/zstd-subbuild/zstd-populate-prefix/src/zstd-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
