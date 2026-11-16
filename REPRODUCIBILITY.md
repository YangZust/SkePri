# Integration and verification notes

Authority: author-supplied `SkePri_PR.pdf`, sections 3.2–3.4 and 4.2, the revised LLM-prior supplementary material, and the original experimental source archives. The two release repair instructions explicitly preserve all 20 formal configurations.

## Completed

- One integrated model file, one train/test entry point, 20 formal YAML files and one four-stream fusion tool. No ablation-specific models or configs, legacy runner copies, datasets or checkpoints.
- Fixed paper-aligned prior interaction, shared D→1 auxiliary head, CE loss, and physically stripped inference state. NTU60 1,446,929 training parameters; 1,446,672 inference parameters. NTU120 1,462,349/1,462,092; UCLA 1,427,149/1,426,892. Each adds exactly 257 training parameters.
- Preserve original NTU PYSKL and UCLA CTR-GCN graph convolution orientations and temporal block implementations within the integrated model.
- Correct the NTU60 X-View JM stream feature selection, which was `bm` in a historical config. All configs now select their named stream.
- Embed the supplied missing-skeleton list in the NTU preparation script and use the source UCLA split metadata. Preserve the four formal prior matrices without changing their values; retain the unused historical bank in the original archive.
- CPU synthetic functional tests passed (see `tests/smoke_test.py`). Synthetic raw NTU60 and NTU120 preprocessing also passed, including official split logic and the embedded 535-ID exclusion list.

## Decisions where manuscript/source leave ambiguity

1. The manuscript omits the person axis. Sample-level node features are constructed by temporal and person averaging before Eq. (6). This differs from legacy per-person auxiliary-logit averaging and is explicitly documented.
2. Formal configs use the paper's default lambda=0.01 and full 90/65 epoch schedules. Old NTU120 experiments used varied auxiliary head designs, weights, and a 30-epoch fine-tuning variant; these are not silently reproduced. The supplied auxiliary versions do not support claiming old accuracy values for this implementation.
3. The formal NTU banks are dataset-level defaults. An alternative NTU120 bone prior remains in the original archive; it was omitted from this release after confirming that no Python or YAML file references it. UCLA source configs specify separate joint and bone banks.
4. Historical fusion files include mixed baseline/SkePri scores (NTU120 X-Set and NW-UCLA), and the final selected weights were not supplied. The unified ensemble requires explicit weights, and the example uses equal weights only for demonstration. Verify final weights and four-stream scores before reporting Table 1 reproduction.
5. The author-supplied original NTU60 `generate_gpt_matrix_api.py` was subsequently located and inspected statically. Its exact prompts, original class ordering and joint-string formatting are released in `priors/prompts/ntu60.json`; connection configuration was not copied. The original NTU120 and NW-UCLA generation protocols have not been validated for this release. Re-requesting GPT-4o cannot guarantee identical matrices; the released numerical banks remain authoritative for reproducing their values.
6. Official test-split checkpoint selection is intentionally retained, per paper. It is not independent validation. Full-data training, reported Top-1 values, the original CUDA 11.3 / RTX 4090 environment and actual inference FLOP profiling have not been verified here. The local synthetic GPU checks described below, matched parameter counts and inference output equality are functional checks, not proof of accuracy reproduction.
7. Source licenses are retained. CTR-GCN's noncommercial license must be considered before choosing the public release license for this derivative repository.

8. All formal configurations now use five training dataset repetitions per epoch, as explicitly requested for this release. This replaces the previous single repetition in NTU60 X-View, NTU120 X-Sub and NTU120 X-Set. Validation/test repetition remains one, with unchanged clip settings. The existing epoch schedules and all other hyperparameters are retained; historical accuracy results are not evidence of accuracy under the revised training schedule.

## Verification command

`python tests/smoke_test.py` (no benchmark data necessary). Full dataset preparation and benchmark runs must be performed in the target GPU environment.

## Environment compatibility check (2026-11-16)

The original experiment environment was confirmed by the authors using commands on their server: Python 3.7.11, PyTorch 1.11.0, CUDA 11.3, NumPy 1.20.3, PyYAML 6.0, tqdm 4.65.0, and NVIDIA GeForce RTX 4090. This is distinct from the local engineering tests below.

| Local test | Python | PyTorch | NumPy | PyYAML | tqdm | Device / result |
| --- | --- | --- | --- | --- | --- | --- |
| Legacy runtime | 3.7.11 | 1.11.0+cpu | 1.20.3 | 6.0 | 4.65.0 | Windows x64, CPU; full smoke test passed, exit 0 |
| Modern runtime | 3.9.24 | 2.7.1+cu128 | 2.0.2 | 6.0.3 | 4.67.1 | Windows x64, CPU; full smoke test passed, exit 0 |
| Modern GPU | 3.9.24 | 2.7.1+cu128 | 2.0.2 | 6.0.3 | 4.67.1 | CUDA 12.8, RTX 5060 Laptop GPU; synthetic checks passed, exit 0 |

The legacy runtime was isolated from the existing installations. It uses the exact original Python and core library versions, with a CPU-only PyTorch build. It does not validate the original CUDA 11.3 build or RTX 4090 hardware. Neither existing PyTorch/CUDA installations nor benchmark datasets were changed.

Both CPU runs executed `python tests/smoke_test.py` and printed `ALL SMOKE TESTS PASSED`. Coverage includes:

- Checkpoint loading: modern restricted loading, rejection of untrusted legacy loads before deserialization, explicit trusted legacy loads, and propagation of restricted-loader failures without an unsafe fallback. Actual checkpoint round trips are included.
- Synthetic raw NTU60 and NTU120 preprocessing through the CLI, missing-skeleton exclusion, action limits and official split construction.
- All 20 YAML configurations, five training repetitions and auxiliary loss weight 0.01, numerical auxiliary-formula agreement, fixed priors and the 257 training-only parameters.
- Forward/backward computation for all three datasets, auxiliary and backbone gradients, and identical inference outputs after auxiliary removal.
- A synthetic one-epoch NTU training run, best-checkpoint saving and reloading, deployment-weight export, final ten-clip evaluation, and a separate test-CLI invocation producing identical scores.
- All four NW-UCLA feeder streams and train/evaluation repetition, plus four-stream fusion through both the function and CLI with deliberately reordered sample IDs.

The modern GPU check separately exercised synthetic forward/backward computation, an optimizer step, checkpoint save/load, and deployment-output equality for NTU60, NTU120 and NW-UCLA. It was not a full GPU benchmark run. Synthetic accuracy values in test output are test-fixture results and must not be reported as benchmark accuracy.

Python 3.7 grammar checks passed for the project Python files. The `--help` entry points for training/testing, fusion, raw preprocessing and optional prior generation also passed under Python 3.7.11. OpenAI SDK was absent in the legacy runtime, and importing the core entry points did not import it. Real SDK/API calls were not tested.

The compatibility changes centralize checkpoint loading in `checkpoint.py`. Only trusted checkpoints can use the old loader; externally supplied weights require `--trust-checkpoint`. Newer PyTorch always uses `weights_only=True` even when trust is acknowledged. The initial compatibility work preserved all 37 retained protected source files. The subsequent validation repair changes input checks, preprocessing error handling, provenance and the optional generator; the 29 protected model, graph, YAML and formal-prior files remain byte-for-byte identical to commit 65be3cd. The unused historical prior remains in the original archive.

The core requirements accept the original NumPy/PyYAML/tqdm versions and do not install PyTorch or CUDA. The optional SDK dependency is isolated in `requirements-priors.txt` for a separate Python 3.9+ API-tool environment. The engineering checks above do not establish full-data accuracy reproduction or compatibility with every intermediate library version.

## Validation and supplementary-material repair (2026-11-16)

- Starting commit: 65be3cd; repair branch: fix/release-validation. Model, graph, all 20 YAML and four formal prior files are protected by pre/post SHA-256 checks. The embedded missing-skeleton list and official subject IDs are unchanged. NTU uses the existing epoch-level cosine schedule; UCLA retains five warm-up epochs and its existing epoch-50 step convention. Every training epoch still traverses the training dataset five times; auxiliary weight remains 0.01.
- Score files now carry method, original training seed, selected checkpoint epoch, score_type, schema_version and deployment-checkpoint SHA-256. The deployment .pth remains a pure main-path state_dict; its .meta.json sidecar records provenance. Test-time configuration must match the recorded dataset/protocol/stream. A missing legacy source requires explicit user declaration, with unknown seed/epoch represented by -1. Legacy NPZ scores without provenance are rejected and must be regenerated. Mixed SkePri/Baseline streams are rejected; per-stream seed/epoch differences are retained. Fusion remains ID-aligned weighted tensordot without extra Softmax or weight search.
- Formal NTU completeness is fail-closed: full class coverage and exact sample/label/split agreement with an explicitly supplied trusted official PYSKL reference are required. No independent full reference manifest or verified post-filter counts have been supplied to this repair; no real dataset was certified here. Debug partial-data mode skips certification only, not malformed-input checks. Coordinate checks are lazy in the loader. No real dataset was downloaded or uploaded.
- The retained NW-UCLA built-in lists contain 1020 training and 464 test records (1484 total); their contents and assignment are unchanged. Unsupported normalization=True now errors; only the duplicated assignment was removed, with random-number consumption unchanged. Exact old/new output equality was checked for both splits and all four streams for NTU and UCLA using identical seeds and valid synthetic inputs.
- Prior audit confirms exact duplicates at zero-based NTU120 indices (7,15), i.e. classes (8,16), and NW-UCLA Bone indices (4,9), i.e. classes (5,10). NTU60 and UCLA Joint have no exact duplicate class matrices. These findings are reported only; no prior values or ordering were changed.
- The full smoke test passed with both Python 3.7.11 / torch 1.11.0+cpu / NumPy 1.20.3 and Python 3.9.24 / torch 2.7.1+cu128 / NumPy 2.0.2. Coverage now includes malformed labels/IDs/scores/weights, mixed provenance, both model training lifecycles, checkpoint sidecars/hashes, independent test provenance, config rejection and malformed data. Four independent prior-generation unittest groups passed on both runtimes, using mocked Chat Completions only. No API request, charge or prior regeneration occurred.
- An actual exact-dimension GPU probe used [64,2,100,25,3], NTU120, 120 priors and the unchanged formal model. On RTX 5060 Laptop GPU / torch 2.7.1+cu128 / CUDA 12.8, forward, finite loss, backward and inference passed. PyTorch peak allocated bytes: 13050125312; peak reserved bytes: 14090764288. These are allocator measurements under Windows (potentially including shared-memory behavior), not proof of a physical-VRAM requirement or an RTX 4090 measurement. The probe uses no real data and establishes no benchmark accuracy. A configurable time limit is provided by tests/gpu_test.py to bound future runs.
- To run the functional checks: python tests/smoke_test.py; python tests/test_prior_generation.py; python tests/gpu_test.py --timeout-seconds 900. The original CUDA 11.3 / RTX 4090 environment, all 20 full-data accuracy runs, historical fusion weights and exact historical API responses remain unverified.
