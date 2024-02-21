import comet_ml

stale_experiment = comet_ml.get_global_experiment()
print(stale_experiment)