# After spending time exploring directly reading the original, possibly compressed Parquet file of the MNIST dataset,
# I found it was more complicated than anticipated, therefore, I decided to use `datasets` to read the original
# dataset and convert it to a simpler format.
# This script handles the reading and storing of the MNIST dataset to a specified directory.

from pathlib import Path
from typing import Iterator, cast

from datasets import DatasetDict, load_dataset
from PIL.Image import Image

IMAGE_SIZE = 28 * 28  # Each MNIST image is 28x28 pixels


def read_dataset(input_path: Path) -> DatasetDict:
    dataset = load_dataset(input_path.as_posix())
    dataset = cast(DatasetDict, dataset)
    print(f"Dataset loaded with splits: {dataset.keys()}")

    return dataset


def write_dataset(iterator: Iterator, image_path: Path, label_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)

    with image_path.open("wb") as f_image, label_path.open("w") as f_label:
        for example in iterator:
            image: Image = example["image"]
            label = example["label"]

            f_image.write(image.tobytes())
            f_label.write(f"{label}\n")


def check_file_sizes(image_path: Path, label_path: Path, example_number: int) -> None:
    with image_path.open("rb") as f_image:
        image_data = f_image.read()

        assert (
            len(image_data) == example_number * IMAGE_SIZE
        ), f"Image file size does not match expected size. Expected {example_number * IMAGE_SIZE}, got {len(image_data)}"

    with label_path.open("r") as f_label:
        labels = f_label.readlines()

        assert (
            len(labels) == example_number
        ), f"Label file size does not match expected size. Expected {example_number}, got {len(labels)}"


def process(dataset: DatasetDict, mode: str, project_root: Path) -> None:
    if mode not in ["train", "test"]:
        raise ValueError("Mode must be 'train' or 'test'")

    print(f"Processing {mode} dataset...")

    image_path = project_root / "dataset" / "processed" / mode / "images.bin"
    label_path = project_root / "dataset" / "processed" / mode / "labels.txt"
    example_number = len(dataset[mode])

    write_dataset(
        iter(dataset[mode]),
        image_path,
        label_path,
    )
    check_file_sizes(
        image_path,
        label_path,
        example_number,
    )

    print(
        f"Processed {mode} dataset: {example_number} examples, images at {image_path}, labels at {label_path}"
    )


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.resolve()
    mnist_dataset = read_dataset(project_root / "dataset" / "raw" / "mnist")

    process(mnist_dataset, "train", project_root)
    process(mnist_dataset, "test", project_root)
