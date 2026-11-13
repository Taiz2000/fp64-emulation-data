// dgemm_bench: one GEMM configuration per process, one CSV row out.
//
// C(m x n) = alpha * op(A)(m x k) * op(B)(k x n) + beta * C0, column-major (default N/N, alpha=1, beta=0).
// v5 adds --opa/--opb N|T, --alpha/--beta, --inject <special values>, --batch (strided batched), loadavg.
// Inputs are generated as the logical op(A)/op(B) (so --dist refers to the k index regardless of
// transposition) and stored transposed on the device when op = T.
// Env vars (CUBLAS_EMULATE_DOUBLE_PRECISION, CUBLAS_EMULATION_STRATEGY, ...) are set by the
// caller and only recorded here; --api / --strategy select the in-code variant.
//
// Per run:
//   1. Deterministic inputs from --seed (identical across processes -> output hashes comparable).
//   2. Cold call (timed separately), warm-up, timed reps (CUDA events per call).
//   3. NVML sampler thread during the timed loop: SM clock, power, temperature, clock-event reasons,
//      plus total-energy counter delta.
//   4. Native reference in-process via CUBLAS_COMPUTE_64F_PEDANTIC on a separate handle.
//   5. CPU reference on sampled entries: long double dot product, cross-checked with Dot2
//      (Ogita-Rump-Oishi compensated dot product).
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <omp.h>
#include <cupti.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2);} } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr, "cuBLAS error %s at %s:%d\n", cublasGetStatusName(s_), __FILE__, __LINE__); exit(3);} } while (0)

// ---------------------------------------------------------------- options
struct Opt {
  long m = 4096, n = 4096, k = 4096;
  std::string api = "dgemm";      // dgemm | gemmex64f | pedantic | gemmex_emu | mathmode | int8
  std::string strategy = "none";  // none | default | performant | eager   (API call; "none" = don't call)
  std::string mcontrol = "none";  // none | dynamic | fixed
  int max_bits = -1;              // >=0: cublasSetFixedPointEmulationMaxMantissaBitCount
  int bit_offset = INT32_MIN;     // set: cublasSetFixedPointEmulationMantissaBitOffset
  std::string dist = "u11";       // u11: U[-1,1) | ozaki: (U-0.5)*exp(phi*N(0,1))
  double phi = 0.0;
  uint64_t seed = 1;
  int warmup = 2;
  double warmup_s = 1.0;           // also keep warming up until this much GPU time has passed
  int reps = 0;                   // 0 = auto: enough for ~target_s of timed work, clamped [min_reps,max_reps]
  double target_s = 3.0;
  int min_reps = 5, max_reps = 1000;
  long samples = 4096;            // sampled entries for the CPU reference
  long workspace_mb = -1;         // >=0: cublasSetWorkspace with this many MiB
  bool no_ref = false;            // skip in-process native reference
  bool no_cupti = false;          // skip the CUPTI kernel-name path probe (use under nsys)
  std::string out = "";           // CSV path (appends); empty = stdout only
  std::string tag = "";
  std::string times_out = "";     // optional: per-call timed ms, one per line
  bool keep_pool = false;         // set default mempool release threshold to max (cuBLAS internal ws uses cudaMallocAsync)
  bool no_wall = false;           // skip the wall-clock loops
  char opa = 'N', opb = 'N';      // v5
  double alpha = 1.0, beta = 0.0; // v5: beta != 0 -> C0 from the seed, C reset to C0 before every checked call
  std::string inject = "none";    // v5: none|nan|inf|ninf|subnormal|huge|overflow|zeroblock
  long batch = 1;                 // v5: >1 -> cublasDgemmStridedBatched, identical inputs per batch
  std::string kern_out = "";      // v5: write the CUPTI probe's kernels (name, ns) to this file
};
static const char* BENCH_VERSION = "v5";

static void usage() {
  fprintf(stderr,
          "dgemm_bench --m M --n N --k K --api {dgemm,gemmex64f,pedantic,gemmex_emu,mathmode,int8}\n"
          "  [--strategy none|default|performant|eager] [--mcontrol none|dynamic|fixed] [--max-bits B]\n"
          "  [--bit-offset O] [--dist u11|ozaki|rowcol|inner] [--phi P] [--seed S] [--warmup W] [--warmup-s S] [--reps R|0] [--max-reps R]\n"
          "  [--target-s T] [--samples NS] [--workspace-mb MB] [--no-ref] [--out file.csv] [--tag T]\n"
          "  [--opa N|T] [--opb N|T] [--alpha A] [--beta B] [--batch NB]\n"
          "  [--inject none|nan|inf|ninf|subnormal|huge|overflow|zeroblock] [--kern-out file.tsv]\n");
  exit(1);
}

static Opt parse(int argc, char** argv) {
  Opt o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nx = [&]() -> const char* { if (i + 1 >= argc) usage(); return argv[++i]; };
    if (a == "--m") o.m = atol(nx());
    else if (a == "--n") o.n = atol(nx());
    else if (a == "--k") o.k = atol(nx());
    else if (a == "--api") o.api = nx();
    else if (a == "--strategy") o.strategy = nx();
    else if (a == "--mcontrol") o.mcontrol = nx();
    else if (a == "--max-bits") o.max_bits = atoi(nx());
    else if (a == "--bit-offset") o.bit_offset = atoi(nx());
    else if (a == "--dist") o.dist = nx();
    else if (a == "--phi") o.phi = atof(nx());
    else if (a == "--seed") o.seed = strtoull(nx(), nullptr, 10);
    else if (a == "--warmup") o.warmup = atoi(nx());
    else if (a == "--warmup-s") o.warmup_s = atof(nx());
    else if (a == "--max-reps") o.max_reps = atoi(nx());
    else if (a == "--reps") o.reps = atoi(nx());
    else if (a == "--target-s") o.target_s = atof(nx());
    else if (a == "--samples") o.samples = atol(nx());
    else if (a == "--workspace-mb") o.workspace_mb = atol(nx());
    else if (a == "--no-ref") o.no_ref = true;
    else if (a == "--no-cupti") o.no_cupti = true;
    else if (a == "--keep-pool") o.keep_pool = true;
    else if (a == "--no-wall") o.no_wall = true;
    else if (a == "--out") o.out = nx();
    else if (a == "--tag") o.tag = nx();
    else if (a == "--times-out") o.times_out = nx();
    else if (a == "--opa") o.opa = nx()[0];
    else if (a == "--opb") o.opb = nx()[0];
    else if (a == "--alpha") o.alpha = atof(nx());
    else if (a == "--beta") o.beta = atof(nx());
    else if (a == "--inject") o.inject = nx();
    else if (a == "--batch") o.batch = atol(nx());
    else if (a == "--kern-out") o.kern_out = nx();
    else usage();
  }
  if ((o.opa != 'N' && o.opa != 'T') || (o.opb != 'N' && o.opb != 'T') || o.batch < 1) usage();
  if (o.batch > 1 && o.api != "dgemm") { fprintf(stderr, "--batch needs --api dgemm\n"); exit(1); }
  if (o.api == "int8" && (o.opa != 'N' || o.opb != 'N' || o.batch > 1 || o.inject != "none" || o.beta != 0)) {
    fprintf(stderr, "--api int8 supports none of the v5 options\n"); exit(1);
  }
  return o;
}

// ---------------------------------------------------------------- deterministic RNG
static inline uint64_t splitmix64(uint64_t& x) {
  uint64_t z = (x += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
struct Xoshiro {
  uint64_t s[4];
  explicit Xoshiro(uint64_t seed) { for (auto& v : s) v = splitmix64(seed); }
  static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  inline uint64_t next() {
    uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
  }
  inline double u01() { return (next() >> 11) * 0x1.0p-53; }
  inline double normal() {  // Box-Muller
    double u1 = u01(), u2 = u01();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
  }
};

// Column-major rows x cols; each column has its own stream -> parallel and reproducible.
static void fill(std::vector<double>& M, long rows, long cols, int id, const Opt& o) {
  M.resize((size_t)rows * cols);
#pragma omp parallel for schedule(static)
  for (long j = 0; j < cols; ++j) {
    uint64_t s = o.seed * 0x100000001B3ull ^ ((uint64_t)id << 56) ^ (uint64_t)j * 0xD1B54A32D192ED03ull;
    Xoshiro g(splitmix64(s));
    double* col = &M[(size_t)j * rows];
    if (o.dist == "u11" || o.dist == "rowcol" || o.dist == "inner") {
      for (long i = 0; i < rows; ++i) col[i] = 2.0 * g.u01() - 1.0;
    } else if (o.dist == "ozaki") {
      for (long i = 0; i < rows; ++i) { double u = g.u01(), z = g.normal(); col[i] = (u - 0.5) * std::exp(o.phi * z); }
    } else {
      fprintf(stderr, "unknown dist %s\n", o.dist.c_str()); exit(1);
    }
  }
  // Diagonal scaling by 10^(phi*t), t ~ U[-1,1] (phi = decades of half-range).
  //   rowcol: A rows / B columns scaled -> each dot product sees one scale (benign for per-row/col exponents)
  //   inner : A columns / B rows (the k index) scaled independently -> terms within one dot product span
  //           up to 10^(4*phi)
  if (o.dist == "rowcol" || o.dist == "inner") {
    bool by_row = (o.dist == "rowcol") == (id == 0);  // A: rowcol->rows, inner->cols; B: the reverse
    long len = by_row ? rows : cols;
    std::vector<double> f(len);
    Xoshiro g(o.seed * 0x9E3779B97F4A7C15ull ^ (0x5CA1Eull + id));
    for (auto& x : f) x = std::pow(10.0, o.phi * (2.0 * g.u01() - 1.0));
#pragma omp parallel for schedule(static)
    for (long j = 0; j < cols; ++j)
      for (long i = 0; i < rows; ++i) M[(size_t)j * rows + i] *= by_row ? f[i] : f[j];
  }
}

static uint64_t hash64(const void* p, size_t bytes) {
  const uint64_t* w = (const uint64_t*)p;
  size_t nw = bytes / 8;
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < nw; ++i) { h ^= w[i]; h *= 0x100000001b3ull; h ^= h >> 29; }
  return h;
}

// ---------------------------------------------------------------- v5 helpers
// Special-value injection into the logical A (m x k), all on one row so the forced CPU samples cover it.
static long inj_row(long m) { return m / 3; }
static long inj_col_b(long n) { return n / 2; }  // B column used by "overflow"
static void inject(std::vector<double>& A, long m, long k, std::vector<double>& B, long n, const std::string& what) {
  if (what == "none") return;
  const long r = inj_row(m), l0 = std::min<long>(3, k - 1);
  auto a = [&](long i, long l) -> double& { return A[(size_t)l * m + i]; };
  if (what == "nan") a(r, l0) = std::nan("");
  else if (what == "inf") a(r, l0) = INFINITY;
  else if (what == "ninf") a(r, l0) = -INFINITY;
  else if (what == "subnormal") for (long l = 0; l < k; ++l) a(r, l) *= 1e-310;  // row r subnormal
  else if (what == "huge") for (long l = 0; l < k; ++l) a(r, l) *= 1e300;        // row r near DBL_MAX, C row finite
  else if (what == "overflow") {                                                   // only C(r, c) overflows
    for (long l = 0; l < k; ++l) a(r, l) *= 1e200;
    const long c = inj_col_b(n);
    for (long l = 0; l < k; ++l) B[(size_t)c * k + l] *= 1e200;
  } else if (what == "zeroblock") {                                                // first m/4 rows of A zero
    for (long l = 0; l < k; ++l) for (long i = 0; i < std::max(1L, m / 4); ++i) a(i, l) = 0.0;
  } else { fprintf(stderr, "unknown inject %s\n", what.c_str()); exit(1); }
}
// column-major rows x cols -> column-major cols x rows
static std::vector<double> transposed(const std::vector<double>& M, long rows, long cols) {
  std::vector<double> T((size_t)rows * cols);
#pragma omp parallel for schedule(static)
  for (long i = 0; i < rows; ++i)
    for (long j = 0; j < cols; ++j) T[(size_t)i * cols + j] = M[(size_t)j * rows + i];
  return T;
}
static double loadavg1() {
  double v = -1;
  FILE* f = fopen("/proc/loadavg", "r");
  if (f) { if (fscanf(f, "%lf", &v) != 1) v = -1; fclose(f); }
  return v;
}

// ---------------------------------------------------------------- NVML sampler
struct NvmlStats {
  bool ok = false;
  double sm_mean = 0, sm_min = 0, sm_max = 0, pw_mean = 0, temp_max = 0;
  unsigned long long reasons_or = 0;
  long nsamp = 0;
  double energy_j = -1;
  unsigned util_pre = 0, mem_util_pre = 0, other_procs_pre = 0;
};
struct NvmlSampler {
  nvmlDevice_t dev{};
  bool ok = false;
  std::atomic<bool> run{false};
  std::thread th;
  std::vector<unsigned> sm, pw, tp;
  unsigned long long reasons = 0, e0 = 0;
  void init(const char* pci) {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    ok = nvmlDeviceGetHandleByPciBusId_v2(pci, &dev) == NVML_SUCCESS;
  }
  void pre(NvmlStats& st) {
    if (!ok) return;
    nvmlUtilization_t u;
    if (nvmlDeviceGetUtilizationRates(dev, &u) == NVML_SUCCESS) { st.util_pre = u.gpu; st.mem_util_pre = u.memory; }
    unsigned cnt = 64;
    nvmlProcessInfo_t info[64];
    if (nvmlDeviceGetComputeRunningProcesses(dev, &cnt, info) == NVML_SUCCESS) {
      unsigned other = 0;
      for (unsigned i = 0; i < cnt; ++i) if (info[i].pid != (unsigned)getpid()) ++other;
      st.other_procs_pre = other;
    }
  }
  void start() {
    if (!ok) return;
    nvmlDeviceGetTotalEnergyConsumption(dev, &e0);
    run = true;
    th = std::thread([this] {
      while (run) {
        unsigned c, p, t; unsigned long long r;
        if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &c) == NVML_SUCCESS) sm.push_back(c);
        if (nvmlDeviceGetPowerUsage(dev, &p) == NVML_SUCCESS) pw.push_back(p);
        if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &t) == NVML_SUCCESS) tp.push_back(t);
        if (nvmlDeviceGetCurrentClocksEventReasons(dev, &r) == NVML_SUCCESS) reasons |= r;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    });
  }
  void stop(NvmlStats& st) {
    if (!ok) return;
    run = false;
    th.join();
    unsigned long long e1 = 0;
    if (nvmlDeviceGetTotalEnergyConsumption(dev, &e1) == NVML_SUCCESS && e1 >= e0) st.energy_j = (e1 - e0) / 1000.0;
    st.ok = true;
    st.nsamp = (long)sm.size();
    auto mean = [](const std::vector<unsigned>& v) { double s = 0; for (auto x : v) s += x; return v.empty() ? 0 : s / v.size(); };
    st.sm_mean = mean(sm);
    st.pw_mean = mean(pw) / 1000.0;
    if (!sm.empty()) { st.sm_min = *std::min_element(sm.begin(), sm.end()); st.sm_max = *std::max_element(sm.begin(), sm.end()); }
    if (!tp.empty()) st.temp_max = *std::max_element(tp.begin(), tp.end());
    st.reasons_or = reasons;
  }
};

// ---------------------------------------------------------------- CUPTI kernel-name path probe
// One extra call after the timed loop is profiled; kernel names + durations tell which path ran
// (Ozaki-II/int8 kernels vs native FP64 DMMA), including ADP's runtime fallback inside the emulation algo.
static std::vector<std::pair<std::string, uint64_t>> g_kern;
static void CUPTIAPI cupti_req(uint8_t** buf, size_t* size, size_t* maxrec) {
  *size = 8 << 20; *buf = (uint8_t*)aligned_alloc(8, *size); *maxrec = 0;
}
static void CUPTIAPI cupti_done(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
  CUpti_Activity* rec = nullptr;
  while (cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS)
    if (rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || rec->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      auto* k = (CUpti_ActivityKernel13*)rec;
      g_kern.emplace_back(k->name ? k->name : "?", k->end - k->start);
    }
  free(buf);
}
struct KernProbe { bool ok = false; long nkern = 0; double total_ms = 0, top_frac = 0, emu_frac = 0, f64_frac = 0; std::string top; };
static bool is_emu_kernel(const std::string& n) {
  // 13.8: Ozaki-II pipeline (oz2_*) + int8 GEMMs (nvjet *_biu_*); 13.2-13.6: one fused kernel
  // (cublasLt_fused_imma_dgemm_kernel_*)
  return n.find("oz2") != std::string::npos || n.find("biu") != std::string::npos ||
         n.find("emu_batch") != std::string::npos || n.find("fused_imma_dgemm") != std::string::npos;
}
static bool is_f64_gemm_kernel(const std::string& n) {  // native FP64 GEMM (incl. the fallback inside cublasLt_imma_emu)
  if (is_emu_kernel(n)) return false;
  return n.find("d884gemm") != std::string::npos || n.find("dgemm") != std::string::npos ||
         n.find("f64f64") != std::string::npos;
}

// ---------------------------------------------------------------- main
int main(int argc, char** argv) {
  Opt o = parse(argc, argv);
  const long m = o.m, n = o.n, k = o.k;
  const bool is_int8 = o.api == "int8";
  const long nb = o.batch;
  const double load_pre = loadavg1();
  const cublasOperation_t opA = o.opa == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t opB = o.opb == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N;
  const long lda = o.opa == 'T' ? k : m, ldb = o.opb == 'T' ? n : k;  // storage: A is k x m when T, B is n x k
  const bool has_c0 = o.beta != 0.0;

  // ---- device / versions
  int dev = 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  char pci[32];
  CK(cudaDeviceGetPCIBusId(pci, sizeof pci, dev));
  int cbmaj, cbmin, cbpat;
  CB(cublasGetProperty(MAJOR_VERSION, &cbmaj));
  CB(cublasGetProperty(MINOR_VERSION, &cbmin));
  CB(cublasGetProperty(PATCH_LEVEL, &cbpat));
  if (cbmaj < 13 || (cbmaj == 13 && cbmin < 1)) { fprintf(stderr, "cuBLAS %d.%d too old for FP64 emulation\n", cbmaj, cbmin); return 4; }

  NvmlSampler nv;
  nv.init(pci);
  NvmlStats ns;
  nv.pre(ns);

  // ---- handle configuration
  cublasHandle_t h;
  if (o.keep_pool) {
    cudaMemPool_t pool;
    CK(cudaDeviceGetDefaultMemPool(&pool, dev));
    uint64_t thr = UINT64_MAX;
    CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
  }
  CB(cublasCreate(&h));
  if (o.strategy != "none") {
    cublasEmulationStrategy_t s = o.strategy == "eager" ? CUBLAS_EMULATION_STRATEGY_EAGER
                                  : o.strategy == "performant" ? CUBLAS_EMULATION_STRATEGY_PERFORMANT
                                                               : CUBLAS_EMULATION_STRATEGY_DEFAULT;
    CB(cublasSetEmulationStrategy(h, s));
  }
  if (o.mcontrol != "none")
    CB(cublasSetFixedPointEmulationMantissaControl(
        h, o.mcontrol == "fixed" ? CUDA_EMULATION_MANTISSA_CONTROL_FIXED : CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC));
  if (o.max_bits >= 0) CB(cublasSetFixedPointEmulationMaxMantissaBitCount(h, o.max_bits));
  if (o.bit_offset != INT32_MIN) CB(cublasSetFixedPointEmulationMantissaBitOffset(h, o.bit_offset));
  if (o.api == "mathmode") CB(cublasSetMathMode(h, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH));
  void* d_ws = nullptr;
  if (o.workspace_mb >= 0) {
    if (o.workspace_mb > 0) CK(cudaMalloc(&d_ws, (size_t)o.workspace_mb << 20));
    CB(cublasSetWorkspace(h, d_ws, (size_t)o.workspace_mb << 20));
  }
  // Mantissa bit count pointer: managed memory so it works whether cuBLAS writes from host or device.
  const int SENT = -12345;
  int* bitptr = nullptr;
  CK(cudaMallocManaged(&bitptr, sizeof(int)));
  *bitptr = SENT;
  cublasStatus_t bp_st = cublasSetFixedPointEmulationMantissaBitCountPointer(h, bitptr);
  cublasMath_t mm_rb; cublasEmulationStrategy_t st_rb;
  CB(cublasGetMathMode(h, &mm_rb));
  CB(cublasGetEmulationStrategy(h, &st_rb));

  // ---- inputs
  std::vector<double> A, B, C0;
  std::vector<int8_t> A8, B8;
  double *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC0 = nullptr;
  int8_t *dA8 = nullptr, *dB8 = nullptr;
  int32_t* dC32 = nullptr;
  const size_t sA = (size_t)m * k, sB = (size_t)k * n, sC = (size_t)m * n;  // per batch
  if (!is_int8) {
    fill(A, m, k, 0, o);  // logical op(A), m x k
    fill(B, k, n, 1, o);  // logical op(B), k x n
    inject(A, m, k, B, n, o.inject);
    CK(cudaMalloc(&dA, sizeof(double) * sA * nb));
    CK(cudaMalloc(&dB, sizeof(double) * sB * nb));
    CK(cudaMalloc(&dC, sizeof(double) * sC * nb));
    {
      std::vector<double> At, Bt;
      if (o.opa == 'T') At = transposed(A, m, k);
      if (o.opb == 'T') Bt = transposed(B, k, n);
      const double* hA = o.opa == 'T' ? At.data() : A.data();
      const double* hB = o.opb == 'T' ? Bt.data() : B.data();
      for (long b = 0; b < nb; ++b) {
        CK(cudaMemcpy(dA + b * sA, hA, sizeof(double) * sA, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB + b * sB, hB, sizeof(double) * sB, cudaMemcpyHostToDevice));
      }
    }
    if (has_c0) {
      Opt oc = o; oc.dist = "u11";  // C0 is always plain U[-1,1); --dist only shapes A and B
      fill(C0, m, n, 2, oc);
      CK(cudaMalloc(&dC0, sizeof(double) * sC * nb));
      for (long b = 0; b < nb; ++b) CK(cudaMemcpy(dC0 + b * sC, C0.data(), sizeof(double) * sC, cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dC, dC0, sizeof(double) * sC * nb, cudaMemcpyDeviceToDevice));
    }
  } else {
    A8.resize((size_t)k * m); B8.resize((size_t)k * n);
    Xoshiro g(o.seed);
    for (auto& v : A8) v = (int8_t)(g.next() >> 56);
    for (auto& v : B8) v = (int8_t)(g.next() >> 56);
    CK(cudaMalloc(&dA8, k * m)); CK(cudaMalloc(&dB8, k * n)); CK(cudaMalloc(&dC32, sizeof(int32_t) * m * n));
    CK(cudaMemcpy(dA8, A8.data(), k * m, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB8, B8.data(), k * n, cudaMemcpyHostToDevice));
  }

  const double alpha = o.alpha, beta = o.beta;
  const int32_t ione = 1, izero = 0;
  auto call = [&](cublasHandle_t hh, const std::string& api) -> cublasStatus_t {
    if (nb > 1) {  // strided batched: api is dgemm (measured) or pedantic (native reference)
      if (api == "pedantic")
        return cublasGemmStridedBatchedEx(hh, opA, opB, m, n, k, &alpha, dA, CUDA_R_64F, lda, (long long)sA, dB,
                                          CUDA_R_64F, ldb, (long long)sB, &beta, dC, CUDA_R_64F, m, (long long)sC, nb,
                                          CUBLAS_COMPUTE_64F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
      return cublasDgemmStridedBatched(hh, opA, opB, m, n, k, &alpha, dA, lda, (long long)sA, dB, ldb, (long long)sB,
                                       &beta, dC, m, (long long)sC, nb);
    }
    if (api == "dgemm" || api == "mathmode")
      return cublasDgemm(hh, opA, opB, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, m);
    if (api == "gemmex64f" || api == "pedantic" || api == "gemmex_emu") {
      cublasComputeType_t ct = api == "gemmex64f" ? CUBLAS_COMPUTE_64F
                               : api == "pedantic" ? CUBLAS_COMPUTE_64F_PEDANTIC
                                                   : CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT;
      return cublasGemmEx(hh, opA, opB, m, n, k, &alpha, dA, CUDA_R_64F, lda, dB, CUDA_R_64F, ldb, &beta, dC,
                          CUDA_R_64F, m, ct, CUBLAS_GEMM_DEFAULT);
    }
    if (api == "int8")  // TN layout: A stored k x m, B stored k x n
      return cublasGemmEx(hh, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &ione, dA8, CUDA_R_8I, k, dB8, CUDA_R_8I, k, &izero,
                          dC32, CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    fprintf(stderr, "unknown api %s\n", api.c_str()); exit(1);
  };

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto reset_c = [&]() {  // beta != 0 reads C: restore C0 before every call whose output is checked
    if (has_c0) CK(cudaMemcpy(dC, dC0, sizeof(double) * sC * nb, cudaMemcpyDeviceToDevice));
  };
  auto timed_call = [&](float& ms) -> cublasStatus_t {
    CK(cudaEventRecord(e0));
    cublasStatus_t s = call(h, o.api);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    CK(cudaEventElapsedTime(&ms, e0, e1));
    return s;
  };

  // ---- cold call, warm-up, timed loop
  std::string status = "ok";
  float cold_ms = 0, ms = 0;
  cublasStatus_t s = timed_call(cold_ms);
  if (s != CUBLAS_STATUS_SUCCESS) status = std::string("cold:") + cublasGetStatusName(s);
  int bits_cold = *bitptr;
  int nwarm = 0;
  double warm_total = 0;
  while (status == "ok" && (nwarm < o.warmup || warm_total < o.warmup_s * 1000.0)) {
    s = timed_call(ms);
    if (s != CUBLAS_STATUS_SUCCESS) status = std::string("warm:") + cublasGetStatusName(s);
    warm_total += ms; ++nwarm;
  }
  int reps = o.reps;
  if (reps <= 0) reps = std::max(o.min_reps, std::min(o.max_reps, (int)std::ceil(o.target_s * 1000.0 / std::max(ms, 1e-3f))));
  std::vector<float> t(reps, 0.f);
  int bits_last = SENT;
  KernProbe kp;
  if (status == "ok") {
    nv.start();
    for (int i = 0; i < reps; ++i) {
      if (i == reps - 1) *bitptr = SENT;
      s = timed_call(t[i]);
      if (s != CUBLAS_STATUS_SUCCESS) { status = std::string("timed:") + cublasGetStatusName(s); break; }
    }
    nv.stop(ns);
    bits_last = *bitptr;
  }
  std::vector<float> ts = t;
  std::sort(ts.begin(), ts.end());
  double med = reps % 2 ? ts[reps / 2] : 0.5 * (ts[reps / 2 - 1] + ts[reps / 2]);
  double mean = 0, var = 0;
  for (float x : t) mean += x;
  mean /= reps;
  for (float x : t) var += (x - mean) * (x - mean);
  double cv = reps > 1 ? std::sqrt(var / (reps - 1)) / mean : 0;
  double p10 = ts[(size_t)(0.1 * (reps - 1))], p90 = ts[(size_t)(0.9 * (reps - 1))], tmax = ts[reps - 1];
  if (!o.times_out.empty()) {
    FILE* tf = fopen(o.times_out.c_str(), "w");
    if (tf) { for (float x : t) fprintf(tf, "%.5f\n", x); fclose(tf); }
  }
  double flops = 2.0 * m * n * k * nb;
  double tflops = flops / (med * 1e-3) / 1e12;

  // ---- wall-clock timing (host clock), as an application would see it:
  //   wall_sync: cudaDeviceSynchronize after every call;  wall_pipe: calls back to back, one final sync.
  // CUDA events only see GPU time and miss host-side costs (e.g. async mempool release/remap at sync points).
  double wall_sync = -1, wall_pipe = -1;
  int wreps = std::max(3, std::min(reps, (int)std::ceil(1500.0 / std::max(med, 1e-3))));
  if (!o.no_wall && status == "ok") {
    using clk = std::chrono::steady_clock;
    CK(cudaDeviceSynchronize());
    auto t0 = clk::now();
    for (int i = 0; i < wreps; ++i) { call(h, o.api); CK(cudaDeviceSynchronize()); }
    wall_sync = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / wreps;
    t0 = clk::now();
    for (int i = 0; i < wreps; ++i) call(h, o.api);
    CK(cudaDeviceSynchronize());
    wall_pipe = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / wreps;
  }

  // ---- CUPTI path probe: one extra profiled call
  if (!o.no_cupti && status == "ok") {
    CK(cudaDeviceSynchronize());
    if (cuptiActivityRegisterCallbacks(cupti_req, cupti_done) == CUPTI_SUCCESS &&
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) == CUPTI_SUCCESS) {
      call(h, o.api);
      CK(cudaDeviceSynchronize());
      cuptiActivityFlushAll(1);
      cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
      cuptiActivityFlushAll(1);
      if (!o.kern_out.empty()) {
        FILE* kf = fopen(o.kern_out.c_str(), "w");
        if (kf) {
          for (auto& kv : g_kern) fprintf(kf, "%llu\t%s\n", (unsigned long long)kv.second, kv.first.c_str());
          fclose(kf);
        }
      }
      std::vector<std::pair<std::string, uint64_t>> agg;
      uint64_t tot = 0, emu = 0, f64 = 0;
      for (auto& kv : g_kern) {
        const std::string& nm = kv.first;
        uint64_t d = kv.second;
        tot += d;
        if (is_emu_kernel(nm)) emu += d; else if (is_f64_gemm_kernel(nm)) f64 += d;
        auto it = std::find_if(agg.begin(), agg.end(), [&](auto& x) { return x.first == nm; });
        if (it == agg.end()) agg.emplace_back(nm, d); else it->second += d;
      }
      if (tot > 0) {
        auto top = std::max_element(agg.begin(), agg.end(), [](auto& a, auto& b) { return a.second < b.second; });
        kp.ok = true; kp.nkern = (long)g_kern.size(); kp.total_ms = tot / 1e6;
        kp.top_frac = (double)top->second / tot; kp.emu_frac = (double)emu / tot; kp.f64_frac = (double)f64 / tot;
        kp.top = top->first.substr(0, 90);
        for (auto& c : kp.top) if (c == ',' || c == '"') c = ';';
      }
    }
  }

  // ---- output hash + determinism (re-run once, compare)
  uint64_t hC = 0, hC2 = 0;
  long n_diff = -1;
  double diff_norm = -1, ref_ms = -1;
  double err_comp_max = -1, err_comp_med = -1, err_rel_max = -1;
  double nerr_comp_max = -1, nerr_comp_med = -1, nerr_rel_max = -1, ref_disagree = -1;
  uint64_t hN = 0;
  long nnf = -1, nnf_nat = -1, ncls = -1, batch_mismatch = -1;  // v5
  if (!is_int8 && status == "ok") {
    const size_t sCall = sC * nb;
    // beta = 0: C already holds the last call's output (as in v4). beta != 0: earlier calls accumulated into C,
    // so produce one clean C = alpha*AB + beta*C0 first.
    if (has_c0) { reset_c(); CB(call(h, o.api)); CK(cudaDeviceSynchronize()); }
    std::vector<double> C(sCall), Cn;
    CK(cudaMemcpy(C.data(), dC, sizeof(double) * sCall, cudaMemcpyDeviceToHost));
    hC = hash64(C.data(), C.size() * 8);
    reset_c();
    CB(call(h, o.api));
    CK(cudaDeviceSynchronize());
    {
      std::vector<double> C2(sCall);
      CK(cudaMemcpy(C2.data(), dC, sizeof(double) * sCall, cudaMemcpyDeviceToHost));
      hC2 = hash64(C2.data(), C2.size() * 8);
    }
    if (nb > 1) {  // identical inputs in every batch -> every batch's output must equal batch 0's
      long bm = 0;
      for (long b = 1; b < nb; ++b) bm += memcmp(C.data(), C.data() + b * sC, sizeof(double) * sC) != 0;
      batch_mismatch = bm;
    }
    nnf = 0;
    for (size_t i = 0; i < sCall; ++i) nnf += !std::isfinite(C[i]);
    // native reference (separate default handle, pedantic compute type)
    if (!o.no_ref) {
      cublasHandle_t hr;
      CB(cublasCreate(&hr));
      reset_c();
      CK(cudaEventRecord(e0));
      CB(call(hr, "pedantic"));
      CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1));
      float f; CK(cudaEventElapsedTime(&f, e0, e1)); ref_ms = f;
      Cn.resize(sCall);
      CK(cudaMemcpy(Cn.data(), dC, sizeof(double) * sCall, cudaMemcpyDeviceToHost));
      hN = hash64(Cn.data(), Cn.size() * 8);
      CB(cublasDestroy(hr));
      long nd = 0; double md = 0, mx = 0;
#pragma omp parallel for reduction(+ : nd) reduction(max : md, mx)
      for (size_t i = 0; i < C.size(); ++i) {
        if (C[i] != Cn[i]) ++nd;
        md = std::max(md, std::fabs(C[i] - Cn[i]));
        mx = std::max(mx, std::fabs(Cn[i]));
      }
      n_diff = nd;
      diff_norm = mx > 0 ? md / mx : md;
      // special-value behaviour vs native: entries whose class (nan / +inf / -inf / finite) differs
      long nn = 0, nc = 0;
      auto cls = [](double x) { return std::isnan(x) ? 0 : std::isinf(x) ? (x > 0 ? 1 : 2) : 3; };
#pragma omp parallel for reduction(+ : nn, nc)
      for (size_t i = 0; i < sCall; ++i) { nn += !std::isfinite(Cn[i]); nc += cls(C[i]) != cls(Cn[i]); }
      nnf_nat = nn;
      ncls = nc;
    }
    // CPU reference on sampled entries
    long ns_ = std::min<long>(o.samples, m * n);
    std::vector<long> si(ns_), sj(ns_);
    Xoshiro g(o.seed ^ 0xABCDEF12345ull);
    for (long q = 0; q < ns_; ++q) { si[q] = g.next() % m; sj[q] = g.next() % n; }
    if (o.inject != "none")  // first quarter of the samples on the injected row (sample 0 = the overflow column)
      for (long q = 0; q < ns_ / 4; ++q) { si[q] = inj_row(m); sj[q] = q == 0 ? inj_col_b(n) : g.next() % n; }
    std::vector<double> ec(ns_), ecn(ns_);
    double erm = 0, nerm = 0, rdis = 0;
#pragma omp parallel for reduction(max : erm, nerm, rdis)
    for (long q = 0; q < ns_; ++q) {
      long i = si[q], j = sj[q];
      long double acc = 0, aabs = 0;
      double p = 0, sc = 0;  // Dot2
      for (long l = 0; l < k; ++l) {
        double a = A[(size_t)l * m + i], b = B[(size_t)j * k + l];
        acc += (long double)a * b;
        aabs += std::fabs((long double)a * b);
        double hp = a * b, rp = std::fma(a, b, -hp);
        double sp = p + hp, bb = sp - p, qq = (p - (sp - bb)) + (hp - bb);
        p = sp; sc += qq + rp;
      }
      long double dot2 = (long double)p + sc;
      if (has_c0) {  // reference = alpha*sum + beta*C0, in long double
        long double c0 = C0[(size_t)j * m + i];
        acc = (long double)alpha * acc + (long double)beta * c0;
        dot2 = (long double)alpha * dot2 + (long double)beta * c0;
        aabs = std::fabs((long double)alpha) * aabs + std::fabs((long double)beta * c0);
      } else if (alpha != 1.0) {
        acc *= alpha; dot2 *= alpha; aabs *= std::fabs((long double)alpha);
      }
      // injection tests: a reference entry that is non-finite in double (NaN/Inf input, or overflow that long
      // double still represents) is judged only by the class comparison against native, not by error
      if (!std::isfinite((double)acc) || !std::isfinite((double)aabs)) { ec[q] = ecn[q] = 0; continue; }
      long double denom = aabs > 0 ? aabs : 1;
      ec[q] = (double)(std::fabs((long double)C[(size_t)j * m + i] - acc) / denom);
      if (acc != 0) erm = std::max(erm, (double)(std::fabs((long double)C[(size_t)j * m + i] - acc) / std::fabs(acc)));
      if (!Cn.empty()) {
        ecn[q] = (double)(std::fabs((long double)Cn[(size_t)j * m + i] - acc) / denom);
        if (acc != 0) nerm = std::max(nerm, (double)(std::fabs((long double)Cn[(size_t)j * m + i] - acc) / std::fabs(acc)));
      }
      rdis = std::max(rdis, (double)(std::fabs(acc - dot2) / denom));
    }
    auto mx_med = [](std::vector<double> v, double& mxv, double& mdv) {
      std::sort(v.begin(), v.end()); mxv = v.back(); mdv = v[v.size() / 2];
    };
    mx_med(ec, err_comp_max, err_comp_med);
    err_rel_max = erm;
    if (!Cn.empty()) { mx_med(ecn, nerr_comp_max, nerr_comp_med); nerr_rel_max = nerm; }
    ref_disagree = rdis;
  }

  // ---- CSV
  auto envs = [](const char* k) { const char* v = getenv(k); return std::string(v ? v : ""); };
  char ts_buf[32];
  time_t now = time(nullptr);
  strftime(ts_buf, sizeof ts_buf, "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
  const char* hdr =
      "timestamp,tag,gpu,pci,cc,cublas,api,strategy_api,mcontrol,max_bits,bit_offset,env_emulate,env_strategy,"
      "env_svmask,env_bits,env_esc,m,n,k,dist,phi,seed,workspace_mb,warmup,reps,cold_ms,median_ms,min_ms,mean_ms,cv,"
      "tflops,math_mode_rb,strategy_rb,bitptr_set,bits_cold,bits_last,hash_C,deterministic,hash_native,"
      "bitwise_equal_native,n_diff_native,diff_norm_native,ref_native_ms,err_comp_max,err_comp_med,err_rel_max,"
      "nat_err_comp_max,nat_err_comp_med,nat_err_rel_max,ref_disagree,util_pre,other_procs_pre,nvml_samples,"
      "sm_mhz_mean,sm_mhz_min,sm_mhz_max,power_w_mean,temp_max,clk_reasons_or,energy_j_timed,status,p10_ms,p90_ms,max_ms,kern_n,kern_ms,kern_emu_frac,kern_f64_frac,kern_top_frac,kern_top,keep_pool,wall_reps,wall_sync_ms,wall_pipe_ms,"
      "bench_version,opa,opb,alpha,beta,inject,batch,n_nonfinite,n_nonfinite_native,n_class_mismatch_native,"
      "batch_mismatch,loadavg_pre,loadavg_post";
  char row[8192];
  snprintf(row, sizeof row,
           "%s,%s,%s,%s,%d.%d,%d.%d.%d,%s,%s,%s,%d,%s,%s,%s,%s,%s,%s,%ld,%ld,%ld,%s,%g,%llu,%ld,%d,%d,%.4f,%.4f,%.4f,"
           "%.4f,%.5f,%.5f,%d,%d,%s,%d,%d,%016llx,%d,%016llx,%d,%ld,%.3e,%.4f,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%u,%u,"
           "%ld,%.0f,%.0f,%.0f,%.1f,%.0f,0x%llx,%.3f,%s,%.4f,%.4f,%.4f,%ld,%.4f,%.4f,%.4f,%.4f,%s,%d,%d,%.4f,%.4f,"
           "%s,%c,%c,%g,%g,%s,%ld,%ld,%ld,%ld,%ld,%.2f,%.2f",
           ts_buf, o.tag.c_str(), prop.name, pci, prop.major, prop.minor, cbmaj, cbmin, cbpat, o.api.c_str(),
           o.strategy.c_str(), o.mcontrol.c_str(), o.max_bits,
           o.bit_offset == INT32_MIN ? "" : std::to_string(o.bit_offset).c_str(),
           envs("CUBLAS_EMULATE_DOUBLE_PRECISION").c_str(), envs("CUBLAS_EMULATION_STRATEGY").c_str(),
           envs("CUBLAS_EMULATION_SPECIAL_VALUES_SUPPORT_MASK").c_str(),
           envs("CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT").c_str(), envs("CUBLAS_ENABLE_EXPERIMENTAL_ESC").c_str(),
           m, n, k, is_int8 ? "int8" : o.dist.c_str(), o.phi, (unsigned long long)o.seed, o.workspace_mb, nwarm, reps,
           cold_ms, med, ts[0], mean, cv, tflops, (int)mm_rb, (int)st_rb, bp_st == CUBLAS_STATUS_SUCCESS ? "1" : "0",
           bits_cold, bits_last, (unsigned long long)hC, hC == hC2 ? 1 : 0, (unsigned long long)hN,
           (!o.no_ref && !is_int8) ? (hC == hN ? 1 : 0) : -1, n_diff, diff_norm, ref_ms, err_comp_max, err_comp_med,
           err_rel_max, nerr_comp_max, nerr_comp_med, nerr_rel_max, ref_disagree, ns.util_pre, ns.other_procs_pre,
           ns.nsamp, ns.sm_mean, ns.sm_min, ns.sm_max, ns.pw_mean, ns.temp_max, ns.reasons_or, ns.energy_j, status.c_str(), p10, p90, tmax,
           kp.ok ? kp.nkern : -1, kp.total_ms, kp.emu_frac, kp.f64_frac, kp.top_frac, kp.top.c_str(),
           o.keep_pool ? 1 : 0, wreps, wall_sync, wall_pipe,
           BENCH_VERSION, o.opa, o.opb, o.alpha, o.beta, o.inject.c_str(), nb, nnf, nnf_nat, ncls, batch_mismatch,
           load_pre, loadavg1());
  printf("%s\n%s\n", hdr, row);
  if (!o.out.empty()) {
    FILE* f = fopen(o.out.c_str(), "a+");
    if (!f) { perror("open out"); return 5; }
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) fprintf(f, "%s\n", hdr);
    fprintf(f, "%s\n", row);
    fclose(f);
  }
  return status == "ok" ? 0 : 6;
}
