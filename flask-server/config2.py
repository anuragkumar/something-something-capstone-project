config = {
  "model_name": "3D_model_9_classes",
  "output_dir": "./data/trained_models/",

  "input_mode": "av",

  "data_folder": "./data/videos/",

  "json_data_val": "./data/files/validation_data_9_classes.json",

  "json_file_labels": "./data/files/labels_9_classes.json",

  "num_workers": 5,

  "num_classes": 9,
  "batch_size": 30,
  "clip_size": 72,

  "nclips_train": 1,
  "nclips_val": 1,

  "upscale_factor_train": 1.4,
  "upscale_factor_eval": 1.0,

  "step_size_train": 1,
  "step_size_val": 1,

  "lr": 0.008,
  "last_lr": 0.00001,
  "momentum": 0.9,
  "weight_decay": 0.00001,
  "num_epochs": -1,
  "print_freq": 100,

  "input_spatial_size": 84,

  "column_units": 512,
  "save_features": True
}