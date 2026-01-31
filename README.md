# Ethics + Employment Survey Study (Python)

## Run
1) Put dataset here: data/raw/dataset.xlsx
2) Install deps: pip install -r requirements.txt
3) Print columns (optional): python scripts/print_columns.py
4) Run pipeline:
   - Target: employed_binary
     python -m src.run --target employed_binary
   - Target: stress_level
     python -m src.run --target stress_level

## Outputs
- outputs/cleaned.csv
- outputs/tables/*.csv
- outputs/figures/*.png
- outputs/metrics/metrics.json
- models/model.joblib
