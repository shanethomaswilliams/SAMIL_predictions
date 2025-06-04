# SAMIL Predictions

## Overview
SAMIL (Self-Attention Multiple Instance Learning) is a deep learning framework for classifying medical imaging studies, particularly TIFF and DICOM files. This repository provides tools to load trained SAMIL models, process imaging datasets, and generate predictions with detailed output formats. The application supports both command-line and graphical user interface (GUI) workflows for ease of use.

## Directory Structure
- `main.py`: Main entry point for running predictions (CLI/GUI).
- `src/`: Source code for model, dataset, and prediction saving utilities.
  - `model.py`: SAMIL model architecture.
  - `tiff_study_dataset.py`: Dataset loader and pre-processing for TIFF/DICOM studies.
  - `save_predictions.py`: Utilities for saving predictions in CSV/JSON.
- `sample_data/`: Example `.npy` files for testing and demonstration.
- `weights/`: Pre-trained SAMIL `.pth` model weights.
- `requirements.txt`: Python dependencies.
- `README.md`: This documentation.

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd SAMIL_predictions
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Additional requirements: `torch`, `numpy`, `pillow`, `pydicom`, `opencv-python`, `tkinter` (for GUI).

## Sample Data and Weights
- **Sample Data**: Place `.npy` files in `sample_data/`. See `sample_data/README_samples.md` for details.
- **Model Weights**: Place `.pth` files in `weights/`. See `weights/README_weights.md` for details.

## How to Run
### Using the GUI
1. Run the main script:
   ```bash
   python main.py
   ```
2. A file dialog will prompt you to select the study directory.
3. The application will load the model, process the data, and save predictions.

### Using the Command Line
You can also specify arguments directly (see `main.py` for options):
```bash
python main.py --study_dir <path_to_study> --checkpoint <path_to_weights>
```

## Model Details
- **Architecture**: See `src/model.py` for the SAMIL model definition.
- **Inputs**: TIFF or DICOM studies, pre-processed to 112x112 and normalized.
- **Outputs**: Class probabilities and predicted class for each study.

## Saving Predictions
- **CSV**: Results saved to `results/predictions.csv`.
- **JSON**: Results saved to `results/predictions.json`.
- Each entry includes study ID, predicted class, and class probabilities.

## Troubleshooting & FAQ
- **Missing dependencies**: Ensure all packages in `requirements.txt` are installed.
- **No GPU**: The code will fall back to CPU if CUDA is unavailable.
- **File not found**: Check that paths to data and weights are correct.

## Contact
For questions or issues, please open an issue on the repository.
