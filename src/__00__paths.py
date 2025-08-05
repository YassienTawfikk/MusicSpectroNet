from pathlib import Path


def get_base_dir():
    """Return the project root by locating the top-level directory containing 'src'."""
    here = Path().resolve()
    for parent in [here] + list(here.parents):
        if (parent / "src").is_dir():
            return parent
    raise RuntimeError("Project root not found — 'src/' directory is missing.")


# Project Base Directory
base_dir = get_base_dir()

# Data Directories
data_dir = base_dir / "data"
raw_data_dir = data_dir / "raw"
processed_data_dir = data_dir / "processed"
processed_image_dir = processed_data_dir / "images_128x128"
curated_data_dir = data_dir / "curated"
train_dir = curated_data_dir / "train"
test_dir = curated_data_dir / "test"

# Output Directories
output_dir = base_dir / "outputs"
model_dir = output_dir / "models"
figures_dir = output_dir / "figures"
docs_dir = output_dir / "docs"

# Directories List
data_dir_list = [raw_data_dir, processed_image_dir, train_dir, test_dir]
output_dir_list = [model_dir, figures_dir, docs_dir]
