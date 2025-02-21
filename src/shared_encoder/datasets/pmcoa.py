"""PMC-OA dataset."""

import json
import os
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import pyarrow as pa
import pyarrow.json as pj
import torch
import torchvision.transforms.v2 as transforms
from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example
from omegaconf import MISSING
from PIL import Image
from pyarrow import csv
from torch.utils.data import Dataset


@external_store(group="datasets", root_dir=os.getenv("PMCOA_ROOT_DIR", MISSING))
class PMCOA(Dataset[Example]):
    """Handles loading and processing of the PMC-OA dataset."""

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "valid", "test"] = "train",
        file_type: str = "jsonl",
        image_key: str = "image",
        caption_key: str = "caption",
        csv_separator: str = ",",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, dict[str, torch.Tensor]]]
        ] = None,
        mask_generator: Optional[
            Callable[
                [Dict[str, torch.Tensor], Any],
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ]
        ] = None,
        image_dir: Optional[str] = None,
        subset_percentage: float = 1.0,
    ) -> None:
        """Initialize the dataset object with file paths and configurations.

        Parameters
        ----------
        root_dir : str
            Directory where the dataset is stored.
        split : str, default="train"
            Split of the dataset (train, valid, test).
        file_type : str, default="jsonl"
            Type of the input file (csv or jsonl).
        img_key : str, default="image"
            Key for images in the CSV/JSONL files.
        caption_key : str, default="caption"
            Key for captions in the CSV/JSONL files.
        csv_separator : str, default=","
            Separator used in CSV files. Not used for JSONL.
        transform : Callable, optional, default=None
            Transform applied to images.
        tokenizer : Callable[[torch.Tensor], Dict[str, torch.Tensor]]
            Text tokenizer.
        mask_generator : Callable[[Dict[str, torch.Tensor], Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]], optional, default=None
            Generator for the mask.
        image_dir : str, optional, default=None
            Directory where images are stored, relative to the root directory.
            If not provided, it is assumed to be `'images'`.
        subset_percentage : float, default=1.0
            Percentage of the dataset to use.
        """  # noqa: W505
        if split not in ["train", "valid", "test"]:
            raise ValueError(
                "Invalid split name. Split must be one of 'train', 'valid', or 'test'."
            )
        if file_type not in ["csv", "jsonl"]:
            raise ValueError(
                "Invalid file type. File type must be one of 'csv' or 'jsonl'."
            )

        self.root_dir = root_dir

        if image_dir is None:
            self.image_dir = "images"
        else:
            self.image_dir = image_dir

        self.split = split
        input_filename = os.path.join(root_dir, f"{self.split}.{file_type}")

        self.image_filenames, self.captions = (
            self._csv_loader(input_filename, image_key, caption_key, csv_separator)
            if file_type == "csv"
            else self._jsonl_loader(input_filename, image_key, caption_key)
        )
        if subset_percentage < 1.0:
            torch.manual_seed(0)
            num_samples = int(len(self.image_filenames) * subset_percentage)
            indices = torch.randperm(len(self.image_filenames))[:num_samples]
            self.image_filenames = self.image_filenames.take(indices.numpy())
            self.captions = self.captions.take(indices.numpy())

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.RGB(),
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
        else:
            self.transform = transform
        self.tokenizer = tokenizer
        self.mask_generator = mask_generator

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.captions)

    def __getitem__(self, idx: int) -> Example:
        """Return items in the dataset."""
        image_path = os.path.join(
            self.root_dir, self.image_dir, self.image_filenames[idx].as_py()
        )

        with Image.open(image_path) as img:
            images = self.transform(img)

        caption = self.captions[idx].as_py()
        example = Example(
            {
                Modalities.RGB.name: images,
                Modalities.TEXT.name: caption,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

        tokens = self.tokenizer(caption) if self.tokenizer is not None else None
        if tokens is not None:
            if isinstance(tokens, dict):  # output of HFTokenizer
                assert Modalities.TEXT.name in tokens, (
                    f"Missing key `{Modalities.TEXT.name}` in tokens."
                )
                example.update(tokens)
            else:
                example[Modalities.TEXT.name] = tokens

        if self.mask_generator is not None and self.tokenizer is not None:
            _, masked_labels, masked_text = self.mask_generator(
                tokens,
                self.tokenizer.tokenizer,  # type: ignore
            )
            example[Modalities.TEXT.mask] = masked_text
            example[Modalities.TEXT.target] = masked_labels

        return example

    def _csv_loader(
        self, input_filename: str, img_key: str, caption_key: str, sep: str
    ) -> Tuple[pa.ChunkedArray, pa.ChunkedArray]:
        """Load images, captions from CSV data."""
        table = csv.read_csv(
            input_filename,
            parse_options=csv.ParseOptions(delimiter=sep, newlines_in_values=True),
        )
        return table[img_key], table[caption_key]

    def _jsonl_loader(
        self, input_filename: str, img_key: str, caption_key: str
    ) -> Tuple[pa.ChunkedArray, pa.ChunkedArray]:
        """Load images, captions from JSON data."""
        parse_options = pj.ParseOptions(newlines_in_values=True)
        table = pj.read_json(input_filename, parse_options=parse_options)
        return table[img_key], table[caption_key]


@external_store(group="datasets", root_dir=os.getenv("PMCOA_ROOT_DIR", MISSING))
class PMC2M(Dataset[Example]):
    """PMC-2M dataset.

    Parameters
    ----------
    root_dir : str
        Path to the root folder containing jsonl file with data entries.
    split : {"train", "valid", "test"}
        Dataset split.
    include_extra: bool, default=True
        Whether or not to include the additional data samples extracted by us
        in October 2024.
    use_full_caption : bool, default=False
        Use full captions or not.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function applied to textual captions.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "valid", "test"] = "train",
        include_extra: bool = True,
        use_full_caption: bool = False,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
    ) -> None:
        """Initialize the dataset."""
        data_path = os.path.join(root_dir, "clean_pmc_oa", f"{split}.jsonl")
        with open(data_path, encoding="utf-8") as file:
            entries = [json.loads(line) for line in file.readlines()]

        # convert relative image paths to absolute paths
        for entry in entries:
            entry["subfig_path"] = os.path.join(root_dir, "images", entry["image"])

        if include_extra:
            data_path = os.path.join(root_dir, "clean_pmc_oa", f"pmc_oa2_{split}.jsonl")
            with open(data_path, encoding="utf-8") as file:
                entries.extend([json.loads(line) for line in file.readlines()])

        self.entries = entries

        self.root_dir = root_dir

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.RGB(),
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
        else:
            self.transform = transform

        self.tokenizer = tokenizer
        self.use_full_caption = use_full_caption

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample."""
        entry = self.entries[idx]
        subfig_path = os.path.join(self.root_dir, entry["subfig_path"])
        try:
            with Image.open(subfig_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            print(
                f"Error loading image for entry {idx}: image_path={subfig_path}",
                e,
            )
            idx = (idx + 1) % len(self.entries)
            return self.__getitem__(idx)
        if self.use_full_caption:
            caption = entry["full_caption"]
        else:
            caption = entry["sub_caption"]

        if self.transform is not None:
            image = self.transform(image)

        tokens = self.tokenizer(caption) if self.tokenizer is not None else None

        example = Example(
            {
                Modalities.RGB.name: image,
                Modalities.TEXT.name: caption,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

        if tokens is not None:
            if isinstance(tokens, dict):  # output of HFTokenizer
                assert Modalities.TEXT.name in tokens, (
                    f"Missing key `{Modalities.TEXT.name}` in tokens."
                )
                example.update(tokens)
            else:
                example[Modalities.TEXT.name] = tokens

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.entries)
