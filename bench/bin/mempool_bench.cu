// mempool_bench: cost of cudaMallocAsync/cudaFreeAsync + host sync on the device default mempool, the pattern
// cuBLAS uses for its internal workspace when the user sets none (phase P3b, the zero-code memory penalty).
// Per size S and variant, ITERS iterations of:
//   alloc : cudaMallocAsync(S) [+ memset S if touch] ; cudaFreeAsync ; cudaStreamSynchronize
//   base  : memset on a preallocated buffer [if touch] ; cudaStreamSynchronize            (no allocation)
// with the pool's release threshold 0 (CUDA default: memory returned to the OS at every sync) or UINT64_MAX.
// Prints CSV: gpu,size_mib,threshold,touch,variant,iters,median_ms,min_ms,max_ms
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2);} } while (0)

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 20;
  std::vector<size_t> sizes_mib = {64, 128, 256, 512, 1024, 2048, 4096, 6144};
  for (int i = 2; i < argc; ++i) sizes_mib.push_back(strtoull(argv[i], nullptr, 10));  // extra sizes (cuBLAS ws)
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  cudaMemPool_t pool; CK(cudaDeviceGetDefaultMemPool(&pool, 0));
  cudaStream_t st; CK(cudaStreamCreate(&st));
  size_t free_b, total_b; CK(cudaMemGetInfo(&free_b, &total_b));
  printf("gpu,size_mib,threshold,touch,variant,iters,median_ms,min_ms,max_ms\n");
  using clk = std::chrono::steady_clock;
  for (size_t mib : sizes_mib) {
    size_t S = mib << 20;
    if (S + (512ull << 20) > free_b) { fprintf(stderr, "skip %zu MiB (free %zu MiB)\n", mib, free_b >> 20); continue; }
    for (uint64_t thr : {(uint64_t)0, UINT64_MAX}) {
      CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
      CK(cudaMemPoolTrimTo(pool, 0));
      for (int touch = 0; touch <= 1; ++touch) {
        for (int variant = 0; variant <= 1; ++variant) {  // 0 = alloc, 1 = base
          void* pre = nullptr;
          if (variant == 1) CK(cudaMalloc(&pre, S));
          std::vector<double> t;
          for (int it = 0; it < iters + 2; ++it) {  // 2 warm-up iterations
            auto t0 = clk::now();
            void* p = pre;
            if (variant == 0) CK(cudaMallocAsync(&p, S, st));
            if (touch) CK(cudaMemsetAsync(p, 0, S, st));
            if (variant == 0) CK(cudaFreeAsync(p, st));
            CK(cudaStreamSynchronize(st));
            if (it >= 2) t.push_back(std::chrono::duration<double, std::milli>(clk::now() - t0).count());
          }
          if (pre) CK(cudaFree(pre));
          std::sort(t.begin(), t.end());
          printf("%s,%zu,%s,%d,%s,%d,%.4f,%.4f,%.4f\n", prop.name, mib, thr ? "max" : "0", touch,
                 variant ? "base" : "alloc", iters, t[t.size() / 2], t.front(), t.back());
          fflush(stdout);
        }
      }
    }
  }
  return 0;
}
