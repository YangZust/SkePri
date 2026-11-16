# SkePri release-validation and supplementary-material repair

Date: 2026-11-16. Base: 65be3cd747fa06e98c4ca22ee95719a73bb5b24b (docs/ctr-gcn-style). Repair branch: fix/release-validation. The base matched the fetched remote, and the initial working tree was clean. Both author-requested repair tasks are covered; the supplemental task's model/training restrictions apply to that task and do not undo the separately authorized engineering fixes.

## Evidence and scope

Method authority: the supplied SkePri_PR.pdf, especially Eqs. (4)-(12) and section 4.2. Prompt authority: SkePri_Supplementary_LLM_Prior_Revised.docx and the author-supplied original generate_gpt_matrix_api.py. Neither manuscript nor Word file was modified. No real training data, credentials, source archives, test logs or private paths are included in this commit.

The original prompt script was located in the author's local historical files after no match was found in the two experimental tar archives. It was parsed statically with Python AST, never executed or uploaded. Only public prompt text and generation parameters were extracted. Source SHA-256: bba03b7c4c0b0727386109bc74156d0841602cbfc850441e220598726531e899.

## Engineering findings and fixes

| Location at base | Finding | Repair |
| --- | --- | --- |
| ensemble.py:fuse | labels [N,1] could broadcast against predictions [N]; zip silently truncated wrong file counts | validate_score_file rejects malformed labels, scores, metadata and IDs; exactly four files/weights, finite positive weight sum; ID alignment retained |
| main.py:save_scores/train/test | prediction provenance absent; deploy-only model could not reveal training method | actual training model determines method; best checkpoint metadata, deployment sidecar/hash, score provenance, best_result metadata; independent testing verifies dataset/protocol/stream and hash |
| main.py:test | implicit best_inference fallback and possible invented test-time source | --weights required; legacy method must be declared; absent historical seed/epoch remain -1 |
| data/prepare_ntu.py | empty trajectories/denoising and all-zero sequences could assert or index out of bounds; incomplete datasets accepted | sample/path-specific errors, complete statistics, strict trusted-reference comparison, explicit debug partial mode; original filters/thresholds/splits preserved |
| feeders/feeder_ntu.py | conversion to set hid duplicates; malformed annotations/splits unchecked | eager metadata validation and lazy coordinate checks; dataset passed explicitly; original transform order preserved |
| feeders/feeder_ucla.py | duplicate assignment, unusable normalization=True, unchecked JSON | remove duplicate only, reject unsupported interface, explicit sample/path errors; retain all RNG calls and built-in records |
| main.py:load_config | interval and unknown schedules silently ignored | validate positive integer sizes and known schedules; interval must equal 1 |
| tools/generate_priors.py | user-only requests, no bundled original prompts | structured configuration plus legacy-list compatibility; exact system/user messages; explicit CLI > JSON > defaults; protected output paths and original matrix validation |

### Representative before/after contracts

Before: accuracy comparison could accept labels with shape [N,1]. After:

    if y.ndim != 1 or y.dtype.kind not in 'iu':
        raise ValueError(...)

Before: torch.save contained model, epoch and val_top1 only. After: those same keys remain, plus a plain-data metadata dictionary. best_inference.pth remains a pure main-path state_dict. The corresponding best_inference.meta.json carries schema_version=1, method, dataset, protocol, stream, seed, checkpoint_epoch, score_type and checkpoint_sha256. best_scores.npz stores these as scalar strings/integers, with no object arrays. best_result.json carries the same provenance.

Before: test could infer a deployment-weight filename from work_dir. After: --weights is mandatory. Full checkpoints use embedded metadata; deployment checkpoints use .meta.json; hash and configuration mismatches fail. --baseline cannot override recorded SkePri provenance. Legacy weights require --source-method; optional --source-seed/--source-epoch describe training, not testing. Legacy NPZ files without provenance are rejected with a request to regenerate scores. No unsafe checkpoint fallback was added.

Fusion still computes exactly:

    scores=np.tensordot(weights/weights.sum(),np.stack(arrays),axes=1)

There is no added Softmax, per-stream normalization or weight search. Method/dataset/protocol/score_type must match, but individual stream seeds and epochs may differ. Fused outputs also preserve fusion_weights and JSON-encoded stream_sources as a plain string.

Before: gen_keypoint_array passed an empty filtered list to spatial denoising. After: a ValueError reports length filtering, spatial denoising, no bodies or no valid frames; prepare adds sample ID and file path. Malformed files never silently disappear, even with --allow-partial. Float16 storage, length_threshold=11, wh_ratio=0.8, spnoise_ratio=0.69754, MISSING and TRAIN_SUBJECTS remain unchanged.

### Data completeness limitation

A trustworthy independently verified full PYSKL sample manifest/count record was not supplied. No effective sample counts were invented from raw dataset totals. Formal preprocessing/training defaults to strict class coverage and exact ID/label/train/test-set comparison against the user's explicitly supplied trusted official PKL (--reference for preprocessing, --reference-data for main.py). Without a reference, formal certification fails closed. --allow-partial is debug-only and never skips malformed-input checks. The reference's official provenance must be established by the operator; matching an arbitrary reference alone cannot prove official completeness. No real dataset was certified during this task.

The current NW-UCLA embedded lists were recounted: train 1020, test 464, total 1484. Lists and assignment were preserved, rather than changed to match the paper's raw total of 1494.

## Original NTU60 generation materials

The JSON fields are dataset, model, temperature, system_prompt, user_prompt_template, joint_names, joints_info, class_names, prompts and provenance. There are 25 joints, 60 classes and 60 fully expanded prompts. Original whitespace and joint-string separators are retained. Joint index 20 remains spine (Kinect SpineShoulder). Index 8 uses standing up, index 25 uses hopping (one foot), and index 26 uses jumping up, preserving the historical script rather than replacing these with official standardized spellings. Index i maps to dataset label i, or official action number i+1; no sorting or reordering occurs.

The original System prompt is exactly:

    You are a helpful assistant that strictly outputs raw JSON 2D arrays.

Before: messages=[{role: user, content: prompt}]. After for structured inputs: messages=[{role: system, content: recorded_system}, {role: user, content: recorded_user}]. The original request specifies only model=gpt-4o, temperature=0.1 and messages; no seed, top_p, max_tokens or fixed snapshot is invented. The historical script assembled the bank as float32; the released tool retains float32 storage and rejects invalid matrices without symmetrization, clipping or offline row normalization.

Command (optional generation, not required for training):

    python tools/generate_priors.py --dataset ntu60 --prompts priors/prompts/ntu60.json --output generated_priors/ntu60_gpt4o.npy

Explicit CLI model/temperature overrides JSON values, which override defaults; effective values are printed. Old lists remain user-only custom requests. Structured template-derived prompts are checked against the template. Formal paths are protected even if temporarily absent, and existing files are never overwritten. OpenAI SDK remains a separate Python 3.9+ dependency. No API call, charge or regenerated formal bank was produced. Exact historical responses are not reproducible from a mutable model identifier alone; original NTU120/UCLA generation protocols were not audited in this task.

## Prior audit

All four banks have the required shapes, finite entries in [0,1], unit diagonals, and valid row normalization. Exact duplicate class matrices were confirmed with np.array_equal:

- NTU120: zero-based (7,15), one-based class numbers (8,16).
- NW-UCLA Bone: zero-based (4,9), one-based (5,10).
- NTU60 / NW-UCLA Joint: no exact duplicate class matrices.

No duplicates were removed, modified or perturbed.

## Actual verification

- Baseline before repair: Python 3.7.11 / torch 1.11.0+cpu / NumPy 1.20.3 smoke test passed (exit 0).
- After repair: full smoke tests passed on both that legacy CPU environment and Python 3.9.24 / torch 2.7.1+cu128 / NumPy 2.0.2 (exit 0). Test output ends with ALL SMOKE TESTS PASSED. PyYAML/tqdm in the legacy environment are 6.0/4.65.0.
- Separate prior-generation tests: four unittest groups passed on both runtimes (exit 0), using mocked Chat Completions only. They check exact system/user roles, settings, full mapping, CLI overrides, legacy lists, bad configs, invalid matrices/retry, protected/existing outputs and unchanged formal hashes. SDK installation is not needed for these tests.
- Regression coverage includes score dimensions/types, NaN/Inf, invalid labels/IDs/weights, mismatched stream/sample/protocol/method, legacy NPZ rejection, scalar provenance, both SkePri and Baseline synthetic training/best-checkpoint/deployment/test lifecycles, mismatched sidecar hashes/configurations, legacy checkpoint declarations, all 20 YAML, no-prior inference and fixed auxiliary formulas.
- Data regressions cover empty/short/spatially invalid/all-zero/nonfinite skeletons, annotation and split duplicates/overlap/missing IDs, invalid labels/frame counts, missing/malformed UCLA JSON, unsupported normalization, partial raw datasets and unchanged official exclusions.
- Exact old/new valid-output comparisons passed for both splits and all four streams of NTU and UCLA using identical seeds, plus single- and multi-trajectory denoising comparisons. Random draws were preserved.
- Exact formal-size GPU probe: RTX 5060 Laptop GPU, torch 2.7.1+cu128, CUDA 12.8, [64,2,100,25,3], 120 classes, unchanged current model/prior_chunk and priors. Forward, finite total loss (4.876736164093018), backward and inference passed. Peak allocated bytes 13050125312; peak reserved bytes 14090764288. Windows allocator/shared-memory measurements are not physical-VRAM requirements and are not RTX 4090 results. The future test runner adds a configurable timeout; the successful original probe used the same unchanged model and dimensions.
- All project Python files pass Python 3.7 grammar parsing. All relevant --help entry points pass on Python 3.7. git diff --check is clean.
- No full real-data training, historical accuracy reproduction, original CUDA 11.3 / RTX 4090 retest, optimal fusion-weight search, or real API execution was performed.

## Protected SHA-256 comparison

Before/after equality is checked against the clean base snapshot. All 20 YAML retain every original byte, including train_repeat=5 and aux_weight=0.01. Model, graph, formal priors and training schedules are unchanged. The inherited checkpoint loader, transforms in feeders/tools.py and upstream licenses were also retained.

| File | SHA-256 before and after | Comparison |
| --- | --- | --- |
| config/ntu120-xset/bone.yaml | 2ab5d73ad8c5b5c8400f4c37cc29e16592e10852faced80b3822af57730f47d9 | identical |
| config/ntu120-xset/bone_motion.yaml | 15a482b16d5fd924ed963fdfd491fd82e00280a5fa93e53be47418feece64e0f | identical |
| config/ntu120-xset/joint.yaml | 017ece2f658b43967376e215b4d9730d41106ee123c0408f0e796b05a92f2001 | identical |
| config/ntu120-xset/joint_motion.yaml | 6c704d3703490c3a917515f065ffc598db4fd4bc6865364f2677702cd0789be1 | identical |
| config/ntu120-xsub/bone.yaml | 1f465dcff5e2c44d84ab6f2872934e2ffdb7f56e5c0b52df00fbc15489009fba | identical |
| config/ntu120-xsub/bone_motion.yaml | de1bf8b6fdf7ec562b891a7f60a967295c3605c44c966c6c59963a99aaf75d6d | identical |
| config/ntu120-xsub/joint.yaml | f4fbffb4abe1a1aab6a409b6fc442860c667b9d7b8d6c449ffe0daec25518d10 | identical |
| config/ntu120-xsub/joint_motion.yaml | f8a1435928bbb33a87d5f9268ed0e4b7ff72ad5aae7c2ebfb71635fcf2fd6f70 | identical |
| config/ntu60-xsub/bone.yaml | 9f47df935f730f3fc00d9a3eb445508e9248a14a0b94d148d449d3d6ee08bd43 | identical |
| config/ntu60-xsub/bone_motion.yaml | 5d711716a492250ad7a395961008602265aafb35689ce6655172826258a58f30 | identical |
| config/ntu60-xsub/joint.yaml | 2110568a780d0c25a31e1761d32becc40de3902e9fa22685499cf5e966d71b4b | identical |
| config/ntu60-xsub/joint_motion.yaml | 12fdfb9109f96f8e38fc9fc5ff2dc9200368a06ed586b63015a2fd12df4c81cc | identical |
| config/ntu60-xview/bone.yaml | 388c4d292e063dbd125171f51f6bfbb17b81b7e4b063b18f7eb8471dda7bfad6 | identical |
| config/ntu60-xview/bone_motion.yaml | 9540ed4d24ed2a2e45f6c0edcf1a64c92b9296dc468f569a4db4ac83d45247b0 | identical |
| config/ntu60-xview/joint.yaml | b0cee6471dbe420d510f63729756e09bf74687cbbb1ab29262113964efe93f4e | identical |
| config/ntu60-xview/joint_motion.yaml | a6d6b74563710695a1b91337f857b46ebd7e389bf6db12aef10dabae04065b8c | identical |
| config/nwucla/bone.yaml | d2c2e13e3c5e15782a20fdfbbfc3365c9474914e04d587b6a8b5c24254d4dc3c | identical |
| config/nwucla/bone_motion.yaml | 71b4846642d48987ad9649a4cb282fd006b13223b21ce9826b76543cb54b674d | identical |
| config/nwucla/joint.yaml | bcdde9c07090cefb9e69374f9ad74fc9a4a87d09e2e6ca9883505306fe07c7df | identical |
| config/nwucla/joint_motion.yaml | 9a54afce650d7227353c7d2b8af0c9703d6cc4875eb49a356398dfc5bbed4c24 | identical |
| graph/__init__.py | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | identical |
| graph/ntu_rgb_d.py | b0902ea3bad27c4d6302abd991e2ca95a4976e778975f232722b68aec4123d39 | identical |
| graph/tools.py | 21c4cd4440acd9179ac2c521f29c3498d037bf269550a5411bcda2610b7465e5 | identical |
| graph/ucla.py | a3da8ffa983bf69a0e7efa90552b494c66c0e52467d0cdd8bf1995126d843666 | identical |
| model/skepri.py | 10e3b0d2d294deaafd85bb74b8593babcfd0b197af934af40399861e50eafc16 | identical |
| priors/ntu120.npy | 9bc8fc37f783010375ff644b755b9f40da1e6cabfefaed75ec122e7f75697a53 | identical |
| priors/ntu60.npy | 8f681158a910f0da9345e7c39fe1d5cd0b8d86bad273d424d6218be80dad7757 | identical |
| priors/nwucla_bone.npy | c0214b7586987e97d69dc33b85471196ecfe6a358ea8f7f0db21668dec0e810b | identical |
| priors/nwucla_joint.npy | 7519458405a74237e98995d5d649cb519a3723a63d08038be096377d4a1b23ca | identical |

## Files and final readiness

Modified: README.md, REPRODUCIBILITY.md, data/README.md, data/prepare_ntu.py, ensemble.py, feeders/feeder_ntu.py, feeders/feeder_ucla.py, main.py, priors/README.md, tests/smoke_test.py, tools/generate_priors.py.

Added: validation.py, provenance.py, priors/prompts/ntu60.json, tests/release_checks.py, tests/test_prior_generation.py, tests/gpu_test.py and this report. No source file was deleted. The optional generator changes are the second task; engineering entry-point and feeder changes belong to the first task.

The full textual before/after code diff is available in the branch commit. Run git diff --stat 65be3cd..HEAD for final change totals and git status --short to confirm a clean worktree. Only the dedicated repair branch is pushed; default-branch contents and repository visibility are unchanged. No force push is used.

Readiness: the code is ready for controlled real-data reproduction setup, conditional on the author supplying a trusted full dataset reference, verifying the original target GPU environment, and selecting documented four-stream weights. This is not an unconditional claim of formal benchmark readiness or reproduced paper accuracy. Duplicate priors, historical fusion choices, the explicit multi-person aggregation convention and unavailable audited generation protocols remain author-facing reproducibility issues, not silently repaired algorithms.
