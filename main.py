from wilds import get_dataset

dataset = get_dataset(
    dataset="iwildcam",
    root_dir="src/dataset",
    download=True
)