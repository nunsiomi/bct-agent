# Data Preparation

This folder is where the offline data-preparation notebook lives.

## Workflow

1. Place your data-prep notebook here (e.g. `prepare_data.ipynb`).
2. Run it locally — it should be **idempotent** and **run-once-offline**;
   the Docker services do not execute it.
3. Write any generated artifacts into `data_prep/artifacts/`:
   - cleaned CSVs / Parquet files
   - sentence-transformer embeddings (`.npy` / `.pkl`)
   - retrieval indices
4. The Task B `retrieval_node` reads from `data_prep/artifacts/` at runtime.

## Suggested layout

```
data_prep/
├── README.md
├── prepare_data.ipynb        # <-- your notebook goes here
├── raw/                      # raw input data (gitignored if large)
└── artifacts/                # outputs consumed by the services
```

Keep large binaries out of git — add patterns to `.gitignore` as needed.
