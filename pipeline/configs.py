# ---------- config.py ----------
# Berisi semua konfigurasi dan hyperparameter pipeline.

PIPELINE_NAME = 'sentiment_analysis_pipeline'
DATA_ROOT = 'data'
PIPELINE_ROOT = 'wildan14ar-pipeline'
METADATA_PATH = f"{PIPELINE_ROOT}/metadata.db"
SERVING_MODEL_DIR = f"{PIPELINE_ROOT}/serving_model"

RUNNER = 'DirectRunner'              # Atau 'DataflowRunner'
BEAM_EXPERIMENTS = ['beam_fn_api']

# Hyperparameters untuk training
TRAIN_NUM_STEPS = 2000
EVAL_NUM_STEPS = 500
TUNE_TRAIN_STEPS = 1000
TUNE_EVAL_STEPS = 500

# Threshold evaluasi
BINARY_ACCURACY_THRESHOLD = 0.7
