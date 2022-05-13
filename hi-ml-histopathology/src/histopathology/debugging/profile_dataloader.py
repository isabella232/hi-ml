from pathlib import Path
import torch
import time
import numpy as np

from monai.data import Dataset, load_decathlon_datalist
from histopathology.datasets.base_dataset import SlidesDataset
from monai.transforms.compose import Compose
from monai.transforms.io.dictionary import LoadImaged
from monai.apps.pathology.transforms import TileOnGridd
from monai.data.image_reader import WSIReader
from torch.utils.data import DataLoader
from histopathology.datasets.panda_dataset import PandaDataset
from histopathology.utils.wsi_utils import image_collate


def load_dataset(slides_dataset: SlidesDataset, strat: str = "slides") -> Dataset:
    base_transform = Compose(
        [
            LoadImaged(keys="image", reader=WSIReader, backend="cuCIM", dtype=np.uint8, level=1, image_only=True),
            TileOnGridd(
                keys=["image"],
                tile_count=44,
                tile_size=224,
                random_offset=True,
                background_val=255,
                return_list_of_dicts=True,
            ),
        ]
    )

    if strat == "slides":
        return Dataset(slides_dataset, base_transform)
    elif strat == "slides_list":
        data_list = slides_dataset.dataset_df.to_dict("records")
        for slide in data_list:
            slide["image"] = Path("/tmp/datasets/PANDA") / Path(slide["image"])
        return Dataset(data_list, base_transform)
    elif strat == "monai":
        training_list = load_decathlon_datalist(
            data_list_file_path="/home/t-kbouzid/workspace/repos/hi-ml/hi-ml-histopathology/src/histopathology/"
            "datamodules/csv.json",
            data_list_key="training",
            base_dir="/tmp/datasets/PANDA",
        )
        return Dataset(training_list, base_transform)


def get_dataloader(
    dataset: SlidesDataset, shuffle: bool, batch_size: int = 1, num_workers: int = 2, strat: str = "strat"
) -> DataLoader:
    transformed_slides_dataset = load_dataset(dataset, strat)
    return DataLoader(
        transformed_slides_dataset,
        batch_size=batch_size,
        collate_fn=image_collate,
        shuffle=shuffle,
        num_workers=num_workers,
        multiprocessing_context="spawn",
        pin_memory=True,
    )


def main() -> None:
    slides_dataset = PandaDataset("/tmp/datasets/PANDA")
    num_workers = 2
    batch_size = 1
    strat = "monai"
    dataloader = get_dataloader(
        dataset=slides_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, strat=strat
    )
    prof = torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=4),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            f"/home/t-kbouzid/workspace/repos/hi-ml/hi-ml-histopathology/logs/{strat}_w_{num_workers}_spawn"
        ),
        record_shapes=True,
        with_stack=True,
    )
    prof.start()
    start = time.time()
    acc = 0.0
    N = 22
    for i, batch in enumerate(dataloader):
        if i > N:
            break
        elapsed = time.time() - start
        acc += elapsed
        print(i, batch["image"].shape, elapsed)
        start = time.time()
        prof.step()
    prof.stop()
    print("total time", acc, "avg time", acc / N)


if __name__ == "__main__":
    main()
