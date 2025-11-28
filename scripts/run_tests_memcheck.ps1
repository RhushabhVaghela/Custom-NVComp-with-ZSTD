# PowerShell helper for running memcheck on WSL or Linux
param(
    [string]$pattern = 'test_roundtrip'
    ,[int]$debugKernelVerify = 0
)

# Running memcheck under WSL (recommended on Windows devs)
wsl bash -lc "cd '/mnt/d/Research Experiments/TDPE_and_GPU_loading/NVComp with ZSTD' && ./scripts/run_tests_memcheck.sh $pattern $debugKernelVerify"
