# [clang][CUDA][driver] Assertion `HostOffloadingInputs.size() == 1` after a `ptxas` job fails in a multi-architecture compile

**Repo:** llvm/llvm-project · **Commit:** `b9b5fb9fec7a` · **Build:** assertions enabled · CUDA 13.4 (ptxas V13.4.59)

## What happens

With **two** `--cuda-gpu-arch` values, if `ptxas` fails for one of them the driver
asserts instead of reporting the error:

```
ptxas fatal   : Unknown option '-not-a-real-flag'
clang++: error: ptxas command failed with exit code 255 (use -v to see invocation)
clang++: clang/lib/Driver/ToolChains/Clang.cpp:8334:
  virtual void clang::driver::tools::Clang::ConstructJob(...):
  Assertion `HostOffloadingInputs.size() == 1 && "Only one input expected"' failed.
```

With a single `--cuda-gpu-arch` the same `ptxas` failure is reported and the driver exits
1, which is the expected behaviour.

## Reproducer

An **empty** translation unit is enough. Two independent ways to make one `ptxas` job
fail:

```bash
: > empty.cu

# (a) an option ptxas does not know
clang++ -x cuda --cuda-path=/usr/local/cuda \
        --cuda-gpu-arch=sm_80 --cuda-gpu-arch=sm_90 \
        -Xcuda-ptxas --not-a-real-flag -c -o /dev/null empty.cu

# (b) an architecture the installed ptxas dropped (CUDA 13 removed sm_5x)
clang++ -x cuda --cuda-path=/usr/local/cuda \
        --cuda-gpu-arch=sm_52 --cuda-gpu-arch=sm_90 -c -o /dev/null empty.cu
```

## Expected

Report the `ptxas` failure and exit non-zero. After the first architecture's job fails
the host offloading action is left with no inputs, and the assertion fires while the
driver builds the host job.
