# How to Run Sample Programs

To run `kineto_playground.cpp` in the `sample_programs` folder, you can use the following steps: (Note: scripts below are hard-coded to a specific set of sample programs, you can modify them to work with a different program. TODO: make these scripts more flexible.)

1. `./build-cu.sh`
    - this generates `kplay-cu.o`
2. `./build.sh`
    - this generates binary called `main`
3. Run `./main`
    - runs your code defined in `kineto_playground.cpp`

## ROCm Memcpy-Kind Repro

`kineto_rocm_memcpy_kind_repro.cpp` is a focused AMD/Kineto repro for checking whether ROCm copy activity records preserve the HIP runtime copy direction. It directly runs:

1. `hipMemcpyAsync(..., hipMemcpyHostToDevice, stream)`
2. `hipMemcpyWithStream(..., hipMemcpyDeviceToHost, stream)`

Build it with the AMD Buck modifier so the binary uses the ROCm activity profiler path:

```bash
buck2 build @//mode/opt -m ovr_config//gpu:amd --show-full-output fbcode//kineto/libkineto/sample_programs:kineto_rocm_memcpy_kind_repro
```

Run the built binary on an AMD devgpu host:

```bash
$(buck2 build @//mode/opt -m ovr_config//gpu:amd --show-full-simple-output fbcode//kineto/libkineto/sample_programs:kineto_rocm_memcpy_kind_repro) \
  --bytes 1048576 \
  --iterations 64 \
  --print-limit 40 \
  --trace-path /tmp/kineto_rocm_memcpy_kind_repro.json
```

Expected output includes runtime activities with `hipMemcpyAsync kind=1` and `hipMemcpyWithStream kind=2`, plus matching GPU memcpy summaries for `Memcpy HtoD` and `Memcpy DtoH`. If the ROCm/Kineto direction attribution issue reproduces in this focused case, a runtime copy with kind `1` or `2` will instead pair with a GPU activity summary such as `Memcpy DtoD kind=DtoD`.

## ROCm Stream-Interleave Repro

`kineto_rocm_stream_interleave_repro.cpp` is a focused AMD/Kineto repro for checking Issue C: whether compute kernels from independent host workers collapse onto too few streams or appear spread across unexpected streams. The program creates one host thread and one HIP stream per worker, launches an identifiable kernel family on each worker's stream, then prints Kineto's kernel stream distribution.

Build it with the AMD Buck modifier:

```bash
buck2 build @//mode/opt -m ovr_config//gpu:amd --show-full-output fbcode//kineto/libkineto/sample_programs:kineto_rocm_stream_interleave_repro
```

Run a 4-worker control on an AMD devgpu host:

```bash
$(buck2 build @//mode/opt -m ovr_config//gpu:amd --show-full-simple-output fbcode//kineto/libkineto/sample_programs:kineto_rocm_stream_interleave_repro) \
  --workers 4 \
  --launches 16 \
  --elements 262144 \
  --kernel-iters 512 \
  --print-limit 0 \
  --trace-path /tmp/kineto_rocm_stream_interleave_repro_4w.json
```

Then run an 8-worker stress case:

```bash
$(buck2 build @//mode/opt -m ovr_config//gpu:amd --show-full-simple-output fbcode//kineto/libkineto/sample_programs:kineto_rocm_stream_interleave_repro) \
  --workers 8 \
  --launches 32 \
  --elements 262144 \
  --kernel-iters 512 \
  --print-limit 0 \
  --trace-path /tmp/kineto_rocm_stream_interleave_repro.json
```

Expected healthy output has `workers * launches` Issue C kernel activities and at least one distinct non-zero stream per worker. If Issue C reproduces in this focused case, the summary will show kernel work collapsing onto fewer streams than workers, or one identifiable worker kernel name appearing across multiple streams. On `devgpu015`, the 4-worker control mapped to 4 Kineto streams, while the 8-worker stress case created 8 distinct HIP stream handles but Kineto reported only 4 kernel streams with worker kernel pairs sharing those stream IDs.
