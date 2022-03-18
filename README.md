# pytorch-upper-Maxilla-and-lower-Mandible-jaws-segmentation
pytorch segmentation for  upper (Maxilla) and lower(Mandible) jaws in a DICOM file.  The Dataset is provided as 2D slices from all 3 orthogonal points of view Axial, Coronal, and Sagittal.

![Dataset](view/Dataset.png)

---
## Download open source dataset and explore it!
[Download_jaw_segmentation_DS.ipynb](Download_jaw_segmentation_DS.ipynb) contains the required code for download and inspect the dataset.

---
## Repo Structure
All the code for training and testing found under [source/](source/).

.\
├── *`Download_jaw_segmentation_DS.ipynb`* (Download and explore the dataset)\
├── *`LICENSE`*\
├── *`README.md`*\
├── **source**\
│   ├── *`baseline.py`* (entry for a single experiment run)\
│   ├── *`experiment_handler.py`* (load multiple experiments configs from [experiments.json](source/experiments.json))\
│   ├── *`experiments.json`* (contains configerations for list of experiement that can be run sequentially)\
│   ├── **helpers**\
│   │   └── *`helpers.py`* (contains helper function used for logging and plotting, etc ...)\
│   ├── **learning**\
│   │   ├── *`dataset.py`* (implement the dataset loaders)\
│   │   ├── *`learner.py`* (implement the training, eval, and test functions)\
│   │   ├── *`losses.py`* (implement multiple loss function)\
│   │   ├── *`model.py`* (implment multiple semantic seg models, currently UNET)\
│   │   └── *`utils.py`* (contains all the utils, reading the configs and get dataset transform functions, get model, etc.)\
│   └── *`option.py`* (contains arguments config if you want to pass them through the terminal instead of a json file)\
└── **view**\
    └── *`Dataset.png`*

