# Structural prior banks

All `.npy` files are unchanged copies of matrices from the supplied experimental code archives. Shapes are `[K,V,V]`. Values are finite in [0,1] and diagonal entries equal 1. Class index follows the corresponding dataset's zero-based action labels; joint index follows the dataset's skeleton order.

| File | Shape | Original archive path / use |
|---|---|---|
| ntu60.npy | 60×25×25 | pyskl/gpt_cpr_matrix.npy; NTU60 |
| ntu120.npy | 120×25×25 | pyskl/gpt_cpr_matrix_ntu120.npy; NTU120 default |
| nwucla_joint.npy | 10×20×20 | pyskl/CTR-GCN-main/gpt_joint_prior_bank_nwucla.npy; J/JM |
| nwucla_bone.npy | 10×20×20 | pyskl/CTR-GCN-main/gpt_bone_prior_bank_nwucla.npy; B/BM |

The model row-normalizes these buffers at initialization, not on disk. The deployment checkpoint excludes the prior. No LLM calls are made in training or testing.

These four banks cover all 20 formal configurations: NTU uses the dataset-level bank for all streams; NW-UCLA uses the joint bank for J/JM and the bone bank for B/BM. The unused historical NTU120 bone bank is retained in the original source archive and omitted from this release; no Python or YAML file references it.

## Optional prior generation

The bundled banks are sufficient for training. Only the optional API generator needs the OpenAI SDK and network access. Use a separate Python 3.9+ environment:

```bash
python -m pip install -r requirements.txt -r requirements-priors.txt
python tools/generate_priors.py --dataset ntu60 --prompts priors/prompts/ntu60.json --output generated_priors/ntu60_gpt4o.npy
```

Run these commands from the repository root with `OPENAI_API_KEY` set in the environment. The bundled NTU60 original experimental prompts were extracted statically from the author-supplied original `generate_gpt_matrix_api.py`; that script was not executed or uploaded. Its file hash is recorded in `prompts/ntu60.json`. The JSON contains the exact System prompt, User template, original joints_info string (including whitespace), 25 joint names, 60 class names and 60 fully expanded prompts in zero-based class order. Index 20 retains the historical name `spine`, corresponding to Kinect v2 SpineShoulder. Original class names such as `standing up`, `hopping (one foot)` and `jumping up` are retained rather than replaced by standardized dataset spellings.

The original NTU60 priors were generated using GPT-4o through the Chat Completions API with a temperature of 0.1. The historical script and supplementary material do not explicitly specify top_p, a random seed, a maximum output-token setting or a fixed model snapshot. Independently regenerated matrices therefore are not guaranteed to match the original experimental priors. Reuse the released bank for experiments requiring the manuscript's original prior values.

Structured prompt JSON requests include the recorded System prompt followed by each User prompt. Explicit CLI model/temperature arguments override JSON settings, which override tool defaults; effective values are printed. A legacy list of complete prompts remains supported as a user-only custom protocol and is not equivalent to the original NTU60 requests. Custom structured files can be supplied for other datasets; original NTU120/NW-UCLA generation prompts have not been independently verified or inferred from NTU60.

The generator rejects invalid shapes, nonfinite values, out-of-range entries and non-unit diagonals. Accepted matrices are stored as float32 without symmetrization, clipping or offline normalization. It refuses existing outputs and all formal prior paths, including if a formal file is temporarily absent. Generation is optional; normal training/inference neither imports the SDK nor requires network access. Mock API tests are in `tests/test_prior_generation.py`; no real API requests were performed during this repair.
