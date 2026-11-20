# ASRproject

-11.17: first simple version: only preprocessor.py and train.py 
-11.18: fixed some bugs and add results into the repo(.out)
-11.20: masked padding and add checkpoint for cloud gpu running

## Training and resuming

Both `train.py` and `train_with_limited.py` accept:

```bash
python train.py --epochs 40 --save-every 1
python train.py --resume outputs/baseline_checkpoints/latest.pt --epochs 60
python train_with_limited.py --resume outputs/limited_checkpoints/latest.pt --epochs 60
```


`--project` use your own project path, ensuring data/project/geo contains all the data, models/mhubert-147 includes model and artifacts/processor for processors

`--checkpoint-dir` and `--output-dir` override the checkpoint and final model
locations. 

example project constructure:

speech_recognition/
├── train.py
├── train_with_limited.py
├── training_checkpoint.py
├── data/
│   └── project/
│       ├── geo/
│       │   ├── train.csv
│       │   └── dev.csv
│       └── ... audio files
├── models/
│   └── mhubert-147/
│       └── ...download from mhubert
└── artifacts/
    └── processor/
        └── ...I use the CTC processor, u can replace by yours
