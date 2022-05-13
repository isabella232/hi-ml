from histopathology.datasets.panda_dataset import PandaDataset
import json

dataset = PandaDataset("/tmp/datasets/PANDA")
data: dict = {"training": []}
for i, row in dataset.dataset_df.iterrows():
    data["training"].append({"image": row["image"], "label": row["isup_grade"]})
with open("csv.json", "w") as fp:
    json.dump(data, fp)
