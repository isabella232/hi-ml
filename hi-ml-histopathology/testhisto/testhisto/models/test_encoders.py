#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Callable, Tuple
import numpy as np
import pytest
from pathlib import Path
from torch import Tensor, float32, nn, rand
from torchvision.models import resnet18

from health_azure.utils import CheckpointDownloader, get_workspace, WORKSPACE_CONFIG_JSON, check_config_json, Workspace
from health_ml.utils.checkpoint_utils import LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
from health_ml.utils.common_utils import CHECKPOINT_FOLDER, DEFAULT_AML_UPLOAD_DIR
from histopathology.models.encoders import (TileEncoder, HistoSSLEncoder, ImageNetEncoder,
                                            ImageNetSimCLREncoder, SSLEncoder)
from histopathology.utils.layer_utils import setup_feature_extractor
from testazure.utils_testazure import get_shared_config_json


TILE_SIZE = 224
INPUT_DIMS = (3, TILE_SIZE, TILE_SIZE)
TEST_SSL_RUN_ID = "CRCK_SimCLR_1654677598_49a66020"


def get_supervised_imagenet_encoder() -> TileEncoder:
    return ImageNetEncoder(feature_extraction_model=resnet18, tile_size=TILE_SIZE)


def get_simclr_imagenet_encoder() -> TileEncoder:
    return ImageNetSimCLREncoder(tile_size=TILE_SIZE)


def get_ssl_encoder(download_dir: Path, workspace: Workspace) -> TileEncoder:
    downloader = CheckpointDownloader(aml_workspace=workspace,
                                      run_id=TEST_SSL_RUN_ID,
                                      download_dir=download_dir,
                                      checkpoint_filename=LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX,
                                      remote_checkpoint_dir=Path(f"{DEFAULT_AML_UPLOAD_DIR}/{CHECKPOINT_FOLDER}/"))
    downloader.download_checkpoint_if_necessary()
    return SSLEncoder(pl_checkpoint_path=downloader.local_checkpoint_path, tile_size=TILE_SIZE)


def get_histo_ssl_encoder() -> TileEncoder:
    return HistoSSLEncoder(tile_size=TILE_SIZE)


def _test_encoder(encoder: nn.Module, input_dims: Tuple[int, ...], output_dim: int,
                  batch_size: int = 5) -> None:
    if isinstance(encoder, nn.Module):
        for param_name, param in encoder.named_parameters():
            assert not param.requires_grad, \
                f"Feature extractor has unfrozen parameters: {param_name}"

    images = rand(batch_size, *input_dims, dtype=float32)

    features = encoder(images)
    assert isinstance(features, Tensor)
    assert features.shape == (batch_size, output_dim)


@pytest.mark.parametrize("create_encoder_fn", [get_supervised_imagenet_encoder,
                                               get_simclr_imagenet_encoder,
                                               get_histo_ssl_encoder,
                                               get_ssl_encoder])
def test_encoder(create_encoder_fn: Callable[[], TileEncoder], tmp_path: Path) -> None:
    if create_encoder_fn == get_ssl_encoder:
        download_dir = tmp_path / "ssl_downloaded_weights"
        download_dir.mkdir()
        with check_config_json(tmp_path, shared_config_json=get_shared_config_json()):
            workspace = get_workspace(aml_workspace=None,
                                      workspace_config_path=tmp_path / WORKSPACE_CONFIG_JSON)
        encoder = create_encoder_fn(download_dir=download_dir, workspace=workspace)   # type: ignore
    else:
        encoder = create_encoder_fn()
    _test_encoder(encoder, input_dims=encoder.input_dim, output_dim=encoder.num_encoding)


def _dummy_classifier() -> nn.Module:
    input_size = np.prod(INPUT_DIMS)
    hidden_dim = 10
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1)
    )


@pytest.mark.parametrize('create_classifier_fn', [resnet18, _dummy_classifier])
def test_setup_feature_extractor(create_classifier_fn: Callable[[], nn.Module]) -> None:
    classifier = create_classifier_fn()
    encoder, num_features = setup_feature_extractor(classifier, INPUT_DIMS)
    _test_encoder(encoder, input_dims=INPUT_DIMS, output_dim=num_features)
