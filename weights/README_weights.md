# Model Weights for SAMIL

This folder contains pre-trained SAMIL model weights in `.pth` format.

## Contents
- Example `.pth` files, each representing a trained SAMIL model checkpoint.
- Each checkpoint contains both the raw model (`state_dict`) and the Exponential Moving Average (EMA) model (`ema_state_dict`).

## How to Use
1. Download or copy your trained SAMIL `.pth` files into this directory.
2. When running the application, specify the path to the desired weights file via the command line or GUI dialog.
3. The application will load the model weights for inference as described in the main README.

## File Format
- Each `.pth` file is a PyTorch checkpoint containing at least two keys: `state_dict` and `ema_state_dict`.
- These are loaded automatically by the application (see `main.py`).

## Notes
- Use the provided weights for demonstration or testing. For your own models, ensure the checkpoint format matches the expected structure.
