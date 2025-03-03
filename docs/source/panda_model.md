# DeepMIL model on PANDA dataset

## Background

The repository contains the configuration for training Deep Multiple Instance Learning (DeepMIL) models for ISUP score
prediction on the [PANDA challenge
dataset](https://www.kaggle.com/c/prostate-cancer-grade-assessment?msclkid=63cd71d8cf6511ec8191222e2876cfec).
The models are developed to reproduce the results described in
[Myronenko et al. 2021](<https://link.springer.com/chapter/10.1007/978-3-030-87237-3_32>) and the example in the
[Project-MONAI
tutorial](https://github.com/Project-MONAI/tutorials/blob/master/pathology/multiple_instance_learning/panda_mil_train_evaluate_pytorch_gpu.py).

A ResNet50 encoder that was pre-trained on ImageNet is downloaded on-the-fly (from
[here](https://download.pytorch.org/models/) at the start of the training run.

## Preparations

Please follow the instructions in the [Readme file](../README.md#setting-up-python) to create a Conda environment and
activate it, and the instructions to [set up Azure](../README.md#setting-up-azureml).

## Mount datasets

If you would like to use the models interactively to debug and/or develop, it is necessary to mount the datasets that are
available in Azure. Instructions to prepare and upload the PANDA dataset are [here](public_datasets.md).
"Mounting" here means that the dataset will be loaded on-demand over the network (see also [the
docs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-with-datasets#mount-vs-download)).

You can mount the dataset by executing this script in `<root>/hi-ml-histopathology`:

```shell
python src/histopathology/scripts/mount_azure_dataset.py --dataset_id PANDA
```

After a few seconds, this may bring up a browser to authenticate you in Azure, and let you access the AzureML
workspace that you chose by downloading the `config.json` file. If you get an error message saying that authentication
failed (error message contains "The token is not yet valid (nbf)"), please ensure that your
system's time is set correctly and then try again. On WSL, you can use `sudo hwclock -s`.

Upon success, the script will print out:

```text
Dataset PANDA will be mounted at /tmp/datasets/PANDA.
```

## Running the model as-is

If you have a GPU available, you can run training on that machine, by executing in `<root>/hi-ml-histopathology`:

```shell
conda activate HimlHisto
python ../hi-ml/src/health_ml/runner.py --model histopathology.SlidesPandaImageNetMILBenchmark
```

However, the GPU demand for this model is rather high. We recommend running in AzureML, on a GPU compute cluster. You
can run the training in the cloud by simply appending name of the compute cluster, `--cluster=<your_cluster_name>`. In
addition, you can turn on fine-tuning of the encoder, which will improve the results further:

```shell
conda activate HimlHisto
python ../hi-ml/src/health_ml/runner.py --model histopathology.SlidesPandaImageNetMILBenchmark --is_finetune --cluster=<your_cluster_name>
```

After a few seconds, this may bring up a browser to authenticate you in Azure, and let you access the AzureML
workspace that you chose by downloading the `config.json` file. If you get an error message saying that authentication
failed, "The token is not yet valid (nbf)", please ensure that your
system's time is set correctly (on WSL, use `sudo hwclock -s`) and then try again.

Then the script will output "Successfully queued run number ..." and a line prefixed "Run URL: ...". Open that
URL to view the submitted run in AzureML, view progress, metrics, etc.

## Expected results

The best runs so far uses Transformer Pooling layer similar to the one implemented in Myronenko et. al 2021
(`pool_type=TransformerPoolingBenchmark.__name__`), in combintation with fine-tuning the encoder.

`SlidesPandaImageNetMIL` model trained with fine-tuning, cross validation metrics (mean ± std):

- Validation accuracy: 0.7828 ± 0.0037
- Validation AUROC: 0.9473 ± 0.002
- Validation QWK: 0.8793 ± 0.0074

For internal reference, this was run `HD_0e805b91-319d-4fde-8bc3-1cea3a6d08dd` on `innereye4ws`.

## Model variants

Six different pooling layers can be used by changing parameter `pool_type` in the configuration file, or chosen on the commandline or from CLI. Available pooling layers include:

- `pool_type=AttentionLayer.__name__`: Attention layer from
  [Ilse et al. 2018](<https://arxiv.org/abs/1802.04712?msclkid=2db09d14d12711ecb63134a1dec7b03e>)
- `pool_type=GatedAttentionLayer.__name__`: Gated attention layer from
  [Ilse et al. 2018](<https://arxiv.org/abs/1802.04712?msclkid=2db09d14d12711ecb63134a1dec7b03e>)
- `pool_type=MaxPoolingLayer.__name__`: Max pooling layer returns frequency normalized weights and the maximum feature
  vector over the first axis
- `pool_type=MeanPoolingLayer.__name__`: Mean pooling layer returns uniform weights and the average feature vector over
  the first axis
- `pool_type=TransformerPooling.__name__`: Transformer pooling layer
- `pool_type=TransformerPoolingBenchmark.__name__`: Transformer pooling layer used in
  [Myronenko et al. 2021](<https://link.springer.com/chapter/10.1007/978-3-030-87237-3_32>)

Use the optional runner argument `--mount_in_azureml` to mount the PANDA dataset on AzureML, instead of downloading it.
This will mean that the job starts faster, but may not run at maximum speed because of network bottlenecks.

## Cross-validation

To use cross-validation, supply the additional commandline flag `--crossval_count=5` for 5-fold cross-validation, like:

```shell
python ../hi-ml/src/health_ml/runner.py --model histopathology.SlidesPandaImageNetMILBenchmark --crossval_count=5 --cluster=<your_cluster_name>
```

Cross-validation will start 5 training runs in parallel. For this reason, cross-validation can only be used in AzureML.

To compute aggregated metrics of the hyperdrive run in Azure ML, replace the `run_id` in
`hi-ml-histopathology/src/histopathology/scripts/aggregate_metrics_crossvalidation.py` with the Run ID of the hyperdrive
run, and run the script as follows:

```shell
conda activate HimlHisto
python hi-ml-histopathology/src/histopathology/scripts/aggregate_metrics_crossvalidation.py
```
