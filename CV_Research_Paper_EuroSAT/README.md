# CV_Research_Paper_EuroSAT

Here I uploaded all the outputs of all notebooks I ran, so that you can see the expected cell outputs.

Google drive link for the models and results: https://drive.google.com/drive/folders/1HxGRBqEbCJgV6F5s3J92MtySX6IaZ-gl?usp=sharing

This is the expected layout once the experiment has been run end-to-end:

```
CV_Research_Paper_EuroSAT/
├── Stage 2: baseline models/
│   ├── resnet50/{training_log.csv, best.pth}
│   └── mobilenet_v2/{training_log.csv, best.pth}
├── Stage 3: fine-tuning SHViT/
│   ├── shvit_s1/{training_log.csv, best.pth, checkpoint_*.pth}
│   ├── shvit_s2/{training_log.csv, best.pth, checkpoint_*.pth}
│   ├── shvit_s3/{training_log.csv, best.pth, checkpoint_*.pth}
│   └── shvit_s4/{training_log.csv, best.pth, checkpoint_*.pth}
└── Stage 4: Benchmarking and Demo/
    └── CV_Research_Paper_EuroSAT_analysis.zip/
        ├── eda_outputs/             (eda.py)
        ├── results/                 (evaluate_all.py — per-model results.json + tables)
        ├── figures/                 (make_figures.py — training curves, accuracy bars, …)
        ├── error_analysis_outputs/  (error_analysis.py)
        └── demo_output/             (demo.py)
```

Heavy artifacts (`best.pth`, `checkpoint_*.pth`) are excluded from git via
`.gitignore`; commit selected analysis outputs (figures, results.json,
summary text) explicitly when re-running.
