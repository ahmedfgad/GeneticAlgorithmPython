# Example datasets

The example datasets are not part of PyGAD and are not stored in this repository. They come from third parties and keep their own licenses. To run the examples that use them, download each dataset and put it in this folder using the paths shown below.

These paths are listed in the repository `.gitignore`, so the data stays out of version control if you add it here.

## What each example needs

| Path in this folder | Used by | Dataset |
| --- | --- | --- |
| `Fruit360/` | `examples/nn/extract_features.py` | Fruit-360 |
| `dataset_features.npy` | `examples/nn/example_classification.py`, `examples/gann/example_classification.py`, `examples/KerasGA/image_classification_Dense.py`, `examples/TorchGA/image_classification_Dense.py` | Built from Fruit-360 |
| `outputs.npy` | same as `dataset_features.npy` above | Built from Fruit-360 |
| `dataset_inputs.npy` | `examples/cnn/example_image_classification.py`, `examples/gacnn/example_image_classification.py`, `examples/KerasGA/image_classification_CNN.py`, `examples/TorchGA/image_classification_CNN.py` | Built from Fruit-360 |
| `dataset_outputs.npy` | same as `dataset_inputs.npy` above | Built from Fruit-360 |
| `Fish.csv` | `examples/nn/example_regression_fish.py`, `examples/gann/example_regression_fish.py` | Fish market |
| `Skin_Cancer_Dataset/` | `examples/KerasGA/cancer_dataset.py`, `examples/KerasGA/cancer_dataset_generator.py` | Skin cancer (benign vs malignant) |

## Where to download

### Fruit-360 (`Fruit360/`)

Created by Horea Mureșan and Mihai Oltean. License: CC BY-SA 4.0.

- GitHub: https://github.com/Horea94/Fruit-Images-Dataset
- GitHub (newer home): https://github.com/fruits-360
- Kaggle: https://www.kaggle.com/datasets/moltean/fruits
- Mendeley Data: https://data.mendeley.com/datasets/rp73yg93n8/1

Put the `apple`, `raspberry`, `mango`, and `lemon` class folders under `Fruit360/`, so the structure is `Fruit360/apple/`, `Fruit360/lemon/`, and so on. `extract_features.py` reads these four classes.

### NumPy arrays built from Fruit-360

The four `.npy` files are not separate downloads. They are arrays built from the Fruit-360 images.

- `dataset_features.npy` and `outputs.npy`: regenerate them by running `extract_features.py` after `Fruit360/` is in place. From the `examples/nn/` folder, run `python extract_features.py`. It reads the four fruit classes and writes both files into this folder.
- `dataset_inputs.npy` and `dataset_outputs.npy`: image arrays used by the CNN examples. There is no bundled script that rebuilds them, so restore your own copy or build them from the Fruit-360 images.

### Fish market (`Fish.csv`)

A CSV with the columns `Species`, `Weight`, `Length1`, `Length2`, `Length3`, `Height`, and `Width`.

- Kaggle: https://www.kaggle.com/datasets/vipullrathod/fish-market

Put the file here as `Fish.csv`.

### Skin cancer (`Skin_Cancer_Dataset/`)

The original source of this dataset was not recorded in the repository. The folder layout (a `benign` folder and a `malignant` folder of images) matches the Kaggle "Skin Cancer: Malignant vs. Benign" dataset, but confirm it matches what the examples expect before you rely on it.

- Likely source (verify first): https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign

Put the image folders here as `Skin_Cancer_Dataset/benign/` and `Skin_Cancer_Dataset/malignant/`.

## Travelling salesman notebook

`examples/example_travelling_salesman.ipynb` reads `data/startbucks.csv`. That file is not in the repository and its source was not recorded. Add your own CSV at `examples/data/startbucks.csv` or change the path in the notebook.
