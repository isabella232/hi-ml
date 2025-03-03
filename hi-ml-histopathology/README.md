# Histopathology Models and Workflows

## Getting started

### Setting up Python

For working on the histopathology folder, please create a separate Conda environment.

```shell
cd hi-ml-histopathology
make env
```

You can then activate the environment via `conda activate HimlHisto`. Set VSCode to use this Conda environment, by choosing "Python: Select Interpreter"
from the command palette.

In the event of dependencies update, you can update your conda environment by running the following:

```shell
cd hi-ml-histopathology
make conda_update
```

### Setting up AzureML

In addition, please download an AzureML workspace configuration file for the workspace that you wish to use:

* In the browser, navigate to the workspace in question
* Click on the drop-down menu on upper right of the page, to the left of your account picture.
* Select "Download config file".
* Save that file into the the repository root.

Once that config file is in place, all Python runs that you start inside the `hi-ml-histopathology` folder will automatically use this config file.

## Running histopathology models

To test your setup, please execute in the `hi-ml-histopathology` folder:

```shell
conda activate HimlHisto
python ../hi-ml/src/health_ml/runner.py --model histopathology.TcgaCrckImageNetMIL  --cluster=training-nd24
```

This should start an AzureML job in the AzureML workspace that you configured above via `config.json`. You may need to adjust the name of
the compute cluster (`training-nd24` in the above example).

### Conda environment

If you start your jobs in the `hi-ml-histopathology` folder, they will automatically pick up the Conda environment file that is present in that folder.
If you start your jobs in a different folder, you need to add the `--conda_env` option to point to the file `<repo_root>/hi-ml-histopathology/environment.yml`.

## Running histopathology tests

In the `hi-ml-histopathology` folder, run

```shell
make call_pytest
```

Inside of VSCode, all tests in the repository should be picked up automatically. You can exclude the tests for the `hi-ml` and `hi-ml-azure` packages by
modifying `python.testing.pytestArgs` in the VSCode `.vscode/settings.json` file.

## Tests that require a GPU

The test pipeline for the histopathology folder contains a run of `pytest` on a machine with 2 GPUs. Only tests that are
marked with the `pytest` mark `gpu` are executed on that GPU machine. Note that all tests that bear the `gpu` mark will
_also_ be executed when running on a CPU machine. You need to manually add a `skipif` flag for tests that are meant to
exclusively run on GPU machines. This also helps to ensure that the test suite can pass when executed outside of the
build agents.

* Tests that run only on a CPU machine: Provide no `pytest` marks

```python
def test_my_code() -> None:
    pass
```

* Tests that run on both on a CPU and on a GPU machine: Add `@pytest.mark.gpu`

```python
@pytest.mark.gpu
def test_my_code() -> None:
    pass
```

* Tests that run only on a GPU machine:

```python
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="This test requires a GPU")
@pytest.mark.gpu
def test_my_code() -> None:
    pass
```
