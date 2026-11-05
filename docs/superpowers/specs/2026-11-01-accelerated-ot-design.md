# accelerated-ot design

**Date:** 2026-11-01
**Status:** approved 2026-11-05 (Numba-first rule)

## Goal

Build a Python optimal transport library in the style of
[PythonOT (POT)](https://pythonot.github.io/) that collects partial, sliced,
Gromov-Wasserstein, linear-embedding, and registration methods from earlier
research code in one maintained package. Every method gets its own module
section and its own documentation page, and every algorithm cites its source.

## Non-goals

- Replacing POT as a general-purpose OT library.
- Shipping experiment scripts, datasets, or plotting code from the source repos.
- Porting methods whose code has no clear license.

## Numba-first rule

Every algorithm in the library is written in Numba. This covers EMD, Sinkhorn
and its variants, partial OT, sliced methods, linear embeddings, GW, PGW, MPGW,
fused PGW, barycenters, and registration. Methods are ported directly into
Numba. There is no intermediate NumPy port.

- Each algorithm has an `@njit` kernel that other `@njit` code can call and
  that can run under `prange` over a batch of problems.
- A thin Python function on top of each kernel is the public API. It checks
  inputs, converts types, and carries the docstring.
- The Numba `emd` from step 1 is the shared base. Every method that needs an
  exact OT solve calls it, not `ot.emd`.
- PyTorch is not a dependency. Runtime dependencies are NumPy, Numba, and SciPy.
  POT is used only in tests as a reference.

## Roadmap

Each step gets its own implementation plan.

| Step | Scope | Depends on |
|---|---|---|
| 1 | Numba `emd`: turn the prototype into the library solver, with tests | none |
| 2 | Classical solvers: Sinkhorn family, 1D OT | step 1 |
| 3 | `partial` and `sliced` | step 1 |
| 4 | `linear` (LOT, LOPT, LGW, LPGW) | steps 1 and 3 |
| 5 | `gromov` (GW, PGW, MPGW, fused PGW, barycenters) | steps 1 and 2 |
| 6 | `registration` | steps 3 and 5 |
| Later | Spherical and other lab methods, scope to be decided | steps 1 to 6 |
| Later | Optional GPU backend, only if a measured workload needs it | none |

## Modules and sources (steps 2 to 6)

| Module | Docs pages | Primary references |
|---|---|---|
| `accelerated_ot.partial` | Partial OT (fixed mass and penalized), 1D partial OT | Bai et al., *Sliced Optimal Partial Transport*, CVPR 2023 |
| `accelerated_ot.sliced` | Sliced OT, sliced partial OT | Bai et al., *Sliced Optimal Partial Transport*, CVPR 2023 |
| `accelerated_ot.gromov` | PGW, MPGW, fused PGW, entropic fused PGW, barycenters | Bai et al., *Partial Gromov-Wasserstein Metric*, ICLR; Bai et al., *Fused Partial Gromov-Wasserstein for Structured Objects*, arXiv:2502.09934 |
| `accelerated_ot.linear` | LOT, LOPT, LGW, LMPGW, LPGW | Bai et al., *Linear Optimal Partial Transport Embedding*, ICML 2023; Bai et al., *Linear Partial Gromov-Wasserstein Embedding*, ICLR 2025 |
| `accelerated_ot.registration` | OPT and SOPT with RBF and TPS models, CPD and ICP-FFD baselines | Bai et al., *Partial Transport for Point-Cloud Registration*, arXiv:2309.15787 |

Dependencies point one way: `registration` and `linear` may import `partial`,
`sliced`, and `gromov`, and those three do not import them back. All modules
import the Numba `emd` and do not import POT.

Each docs page contains the problem definition, API reference, a runnable
example, and a references section.

### Source of truth

| Target | Base code | Notes |
|---|---|---|
| `partial` | `SpaOT/lib/opt.py`, `sopt_shape/lib/lib_ot.py` | Remove the duplicated `emd_lp` and cost helpers. |
| `sliced` | `sopt_shape/lib/sliced_opt.py`, `opt1d.cpp` reference | `opt_plans` is defined twice, so confirm which definition the experiments used. |
| `gromov` | Three-way merge of the fused-pgw code, the SpaOT code, and the patched port | Port each documented fix one at a time. Keep `fused_partial_gromov_lambda`. |
| `linear` | `fused-pgw/lib/linear_gromov.py`, `lib_lopt.py` | The fused-pgw copy is newer: it adds the graph loss. |
| `registration` | `sopt_shape/lib/lib_shape.py`, `cpd.py`, `icp_ffd.py` | Split transforms, methods, and baselines. Remove `import *`. `rotation_re_T` is defined twice. |

The Frank-Wolfe argument `Lambda` means `lambda / omega2`, not `lambda`. The
API must state this convention explicitly.

### Excluded

| Code | Reason |
|---|---|
| `HK/`, `fgw/`, `fugw/`, `unbalancedgw/` | No license file in the vendored copy. Use POT or an optional dependency where one exists. |
| Linearized PGW sources without a license | Merge only after the rights holders confirm the license. The same code in the MIT fused-pgw repo may be used. |
| Experiments, data, notebooks, plotting | Not library code. Small examples go in `examples/`. |

### Attribution rules

- Keep the copyright notice of every MIT source in `THIRD_PARTY_NOTICES.md`.
- Code adapted from POT or another project keeps a comment naming the source.
- Every docs page cites the paper that introduced the method.
- Check each reference against the published version before release.
- `lib_ot.py` lists Bernard Schmitzer as an author, so identify that code and
  keep the credit.

### Testing

- Golden tests: on fixed seeds, each ported function matches its source repo.
- Invariant tests: marginals, mass constraints, and non-negativity hold.
- The existing fused PGW port tests carry over, including the `Lambda`
  convention test.
- Cross-checks: where the problem definitions match, compare against POT's
  `partial_wasserstein_1d`, partial GW, and partial fused GW solvers.

## Later: spherical and lab methods

Steps 1 to 6 cover Yikun Bai's own code only. The methods below come from papers with
other authors or from other lab repos, so they wait for a separate decision.
Co-authoring a paper does not settle the license of its repo.

| Method | Usable source | Notes |
|---|---|---|
| Spherical sliced OT, LCOT, LSSOT | POT (MIT) | The original repos have no license file. |
| Expected sliced plans | POT 0.9.7 (MIT) | |
| Stereographic spherical sliced OT | POT 0.9.8dev (MIT) | Not in a PyPI release yet. `mint-vu/s3wd` has no license file. |
| PAWC, sliced spherical partial OT | `mint-vu/Partial_Wasserstein_on_Circles` (MIT) | Copyright Soheil Kolouri. |
| PTLp | None found | |

## Step 1: Numba `emd`

Port the network simplex solver to Numba so it runs inside `njit` and `prange`.
This needs no compiled extension and no GPU.

A feasibility test with POT 0.9.6.post1 and Numba 0.65.1 called the compiled
`EMD_wrap` from `njit` and `prange`. Costs matched `ot.emd` to about 1e-16, and
16 problems of size n=300 took 0.021 s on 4 threads against 0.081 s serially.
That test used a private mangled symbol, so it is not a release path. It shows
that batch workloads benefit from parallelism. A single solve does not get
faster.

Acceptance criteria:

- Relative cost difference from `ot.emd` is below 1e-9, including degenerate
  inputs.
- Dual variables are valid for the same problems.
- A single solve is at most 2 times slower than the C++ solver.

A prototype port of the network simplex meets the speed target so far. On
uniform 2D Gaussian problems a single solve took 1.13 to 1.51 times as long as
`ot.emd` for n=100 to 5000, and the ratio grows with n. Relative cost differences
stayed below 1e-14. The code and logs are in `prototypes/numba_emd/`.

The prototype was also tested end to end. LOPT embeddings and the LOT barycenter
were written fully in `njit` and compared with the existing `lib_lopt` code with
POT kept serial. On 1 thread the Numba version took 1.1 to 1.6 times as long. On
2 threads with `prange` over the measures it was 1.15 to 1.75 times faster.
Outputs matched to 1e-13. This speed is accepted.

Thread settings matter more than the solver:

- Set the Numba pool size with `NUMBA_NUM_THREADS` at launch. On a node that
  reports many cores but allows only a few, `set_num_threads` alone can stall.
- Numba's OpenMP threads keep spinning after a `prange` call. They slow POT's
  solver when both run in one process, so benchmark them in separate processes.
- POT's `numThreads` greater than 1 made small solves 100 to 1000 times slower
  on a 2-CPU node. Library code calls the solver serially and parallelizes over
  problems instead.

If the speed target is not met, compile the C++ solver with a C interface and
call it from Numba through ctypes.

The port is a derivative work. It keeps the LEMON, Bonneel et al., and POT
copyright notices and citations. Confirm the exact license chain from the
source file headers before porting.

## Acceleration choices

Numba is the default because it composes across the call chain, runs batch
solves in parallel, and installs without a compiler. Network simplex depends on
irregular branching, so JAX and PyTorch do not suit it. Dense methods such as
Sinkhorn and GW gradients are also written in Numba, so one backend covers the
whole library. A GPU backend is considered later only if a measured workload
needs it.
