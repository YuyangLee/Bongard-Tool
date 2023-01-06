# README

# Bongard-Tool

## Generating dataset



## Running baselines

First, clone this repository.

```bash
git clone https://github.com/YuyangLee/Bongard-Tool.git
cd Bongard-Tool
```

To get a quick test on these baselines, you can download all images from this link.

```bash
wget https://aidenology-assets.s3.ap-southeast-1.amazonaws.com/dev/bongard-tools/datasets/FuncTools.1.2.Processed.zip -P Dataset/
```

Then, you can generate tasks by running:

```bash
python data/data_process.py --data_root Dataset/ --name_path toolnames/names.1.2.1.json
```

This will automatically read subdirectories of images under `--data_root` and dump task JSON files under `--data_root`.

After the task generation, you should change the configuration files before you run the scripts.

Take CNN-Baseline as an example.

1. Find the configuration file `baselines/configs/configs_V2/train_cnn_shapebd.yaml` and fill in `data_root` with the root directory of your dataset.
2. Run the training script `bash baselines/scripts/run_cnn_model.sh`

 

