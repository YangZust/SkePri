# Dataset preparation

SkePri does not distribute raw or processed benchmark data. Obtain the datasets under their respective terms. Run commands from the repository root.

## NTU RGB+D 60 and 120

1. Request the skeleton-only data from the [official dataset repository](https://github.com/shahroudy/NTURGB-D). NTU60 uses setups S001–S017; NTU120 requires both the original data and setups S018–S032.
2. Extract the `.skeleton` files. Put NTU60 files in `data/NTU60/` and all NTU120 files in `data/NTU120/` (flat directories). For NTU120, merge the original 60-class and extension files into one directory; the data for NTU60 can also be prepared from the combined directory.
3. Run:

```bash
python data/prepare_ntu.py --dataset ntu60 --raw-dir data/NTU60
python data/prepare_ntu.py --dataset ntu120 --raw-dir data/NTU120
```

The PYSKL-derived preprocessing code embeds the missing-skeleton list, filters those IDs, parses and denoises body tracks, produces `M,T,25,3` skeleton arrays and constructs the official cross-subject/cross-view/cross-setup splits. Output files are `data/nturgbd/ntu60_3danno.pkl` and `data/nturgbd/ntu120_3danno.pkl`. Existing output files are not overwritten. To use a preprocessed annotation from PYSKL instead, place the trusted files at the same paths. Source-provided PYSKL download links:

- [NTU60 3D annotations](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl)
- [NTU120 3D annotations](https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl)

Expected structure: `{'split': {protocol+'_train': [sample_id,...], protocol+'_val': [...]}, 'annotations': [{'frame_dir': id, 'keypoint': array[M,T,25,3], 'label': zero_based_int, 'total_frames': T}, ...]}`. Only load pickle files from trusted sources. Use the full datasets, not a subset, for benchmark experiments. Published preprocessing and exclusions may affect sample counts.

NTU uses the original PYSKL order: 3D centering/spine alignment, joint/bone/motion feature generation, uniform temporal sampling, pose decoding and padding to two persons. Each clip has 100 frames. Validation uses one clip; final test averages probabilities from ten clips. The official test split is used for checkpoint selection, matching the manuscript (not an independent validation set).

## Northwestern-UCLA

Obtain NW-UCLA from the dataset authors or the skeleton archive linked in the [original CTR-GCN repository](https://github.com/Uason-Chen/CTR-GCN). The original README links to [all_sqe.zip](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0). Extract under `data/NW-UCLA/all_sqe/` so individual files such as `a05_s04_e02_v03.json` are directly inside that directory. Each JSON contains `skeletons` with shape `[T,20,3]`.

No separate offline preprocessing is required. The original CTR-GCN feeder's sample IDs, class mapping, view split (train 1/2, test 3), augmentation, normalization, 52-frame sampling and bone pairs are retained. The feeder generates J/B/JM/BM on demand. This dataset uses 20 joints and one person.

```text
data/
├── prepare_ntu.py
├── README.md
├── NTU60/                         # supplied by user
├── NTU120/                        # supplied by user
├── nturgbd/
│   ├── ntu60_3danno.pkl             # generated or downloaded
│   └── ntu120_3danno.pkl
└── NW-UCLA/
    └── all_sqe/                   # supplied by user
        └── *.json
```
