# Integration and verification notes

Authority: supplied `SkePri_PR(5).pdf`, sections 3.2–3.4, 4.2, Table 10, and original `SkePri_Release_Source.tar.gz` plus `SkePri_Supplement.tar.gz`.

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
5. The original LLM prompt/generation script was not provided in either server archive. The optional generator validates prompts and matrices but must be given the actual ordered prompts from the supplement; it cannot reconstruct the exact original matrices. The included prior banks are sufficient for training.
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

Python 3.7 grammar checks passed for all 16 project Python files. The `--help` entry points for training/testing, fusion, raw preprocessing and optional prior generation also passed under Python 3.7.11. OpenAI SDK was absent in the legacy runtime, and importing the core entry points did not import it. Actual SDK/API calls were not tested.

The compatibility changes centralize checkpoint loading in `checkpoint.py`. All three original `main.py` loads and the smoke-test load use that helper. Only trusted checkpoints can use the old loader; externally supplied weights require `--trust-checkpoint`. Newer PyTorch always uses `weights_only=True` even when trust is acknowledged. No algorithm, model, formal prior matrix, data transformation, schedule, or formal YAML setting was changed. SHA-256 comparison against `SkePri_OpenSource_Repeat5.zip` confirmed all 37 retained protected model, graph, feeder, YAML, formal prior, preprocessing, fusion and generator files remained byte-for-byte identical. The one unused historical prior was removed only from the release directory and remains in the original archive.

The core requirements accept the original NumPy/PyYAML/tqdm versions and do not install PyTorch or CUDA. The optional SDK dependency is isolated in `requirements-priors.txt` for a separate Python 3.9+ API-tool environment. The engineering checks above do not establish full-data accuracy reproduction or compatibility with every intermediate library version.
