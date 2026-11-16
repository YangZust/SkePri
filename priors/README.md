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
python tools/generate_priors.py --dataset ntu60 --prompts path/to/prompts.json --output path/to/new_bank.npy
```

Run these commands from the repository root with `OPENAI_API_KEY` set in the environment. Supply complete prompts in class-index order. The original prompts are not reconstructed by this utility, and its SDK/API execution is not covered by the core smoke tests. It refuses to overwrite an existing output; do not replace the formal prior banks.
