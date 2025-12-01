@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc -I"include" -I"src" "tests\test_coverage_gaps.cu" "src\cuda_zstd_fse.cu" "src\cuda_zstd_types.cpp" "src\cuda_zstd_utils.cpp" "src\cuda_zstd_utils.cu" "src\cuda_zstd_xxhash.cu" -o test_coverage_gaps.exe -lcudart
