# Sample Data for SAMIL

This folder contains example `.npy` files for testing and demonstration purposes with the SAMIL application.

## Contents
- Example `.npy` files representing pre-processed imaging studies.
- Each file typically contains a NumPy array corresponding to a single study or a batch of images.

## How to Use
1. Place your sample `.npy` files in this directory.
2. When running the application, point the study directory argument or GUI dialog to this folder to use the sample data.
3. The application will load these files for prediction and output results as described in the main README.

## File Format
- Each `.npy` file should contain a NumPy array formatted as expected by the SAMIL model (see `src/tiff_study_dataset.py` for details).
- Ensure the data matches the input shape and normalization described in the documentation.

## Notes
- These files are for demonstration and testing only. For real use, replace with your own data in the same format.
