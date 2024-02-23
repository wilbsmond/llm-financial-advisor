upload_dataset_to_beam:

    `beam volume upload qa_dataset dataset`

dev_train_beam:

    `BEAM_IGNORE_IMPORTS_OFF=true beam run ./training_beam.py:train`

dev_infer_beam:

    `BEAM_IGNORE_IMPORTS_OFF=true beam run ./tools/inference_run.py:infer -d '{"config_file": "configs/dev_inference_config.yaml", "dataset_dir": "./qa_dataset/dataset", "env_file_path": ".env", "model_cache_dir": "./model_cache"}'`