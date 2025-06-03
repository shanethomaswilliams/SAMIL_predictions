import os
import argparse
import tkinter as tk 
from tkinter import filedialog

import torch
import numpy as np
from torch.utils.data import DataLoader

from src.model import SAMIL 
from src.tiff_study_dataset import build_study_loader
from src.save_predictions import save_prediction

def choose_directory_via_gui(prompt: str = "Select the study directory") -> str | None:
    root = tk.Tk()
    root.withdraw()
    root.update() 
    path = filedialog.askdirectory(title=prompt)
    root.destroy()
    return path or None

def load_model_checkpoint(checkpoint_path):
    """
    Load SAMIL model from checkpoint.
    
    Each checkpoint file contains both the raw model ('state_dict') and 
    the EMA model ('ema_state_dict').
    """
    model = SAMIL()
    ema_model = SAMIL()
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    
    return model, ema_model

def format_probabilities(probs):
    """
    Format probability array to 5 decimal places.
    """
    arr = np.array(probs).flatten()
    return [f"{p:.5f}" for p in arr]

def print_predictions(predictions, study_id: str | None = None):
    """
    Printing properly formatted 
    """
    class_names = ['None', 'Moderate', 'Severe']    
    if study_id:
        print(f"\n===== Predictions for {study_id} =====")
    print("\nRaw Model Prediction:")
    print(f"Class: {predictions['raw_pred_class']} ({class_names[predictions['raw_pred_class']]})")
    formatted_raw_probs = format_probabilities(predictions['raw_probs'])
    print(f"Probabilities: [{', '.join(formatted_raw_probs)}]")
    
    print("\nEMA Model Prediction:")
    print(f"Class: {predictions['ema_pred_class']} ({class_names[predictions['ema_pred_class']]})")
    formatted_ema_probs = format_probabilities(predictions['ema_probs'])
    print(f"Probabilities: [{', '.join(formatted_ema_probs)}]")

def predict_single_image(raw_model, ema_model, image_tensor):
    """
    Make prediction for a single image using both raw and EMA models.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    raw_model.eval()
    ema_model.eval()
    
    raw_model = raw_model.to(device)
    ema_model = ema_model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        raw_outputs, _ = raw_model(image_tensor)
        raw_probs = torch.nn.functional.softmax(raw_outputs, dim=1)
        raw_pred_class = torch.argmax(raw_probs, dim=1).item()
        
        ema_outputs, _ = ema_model(image_tensor)
        ema_probs = torch.nn.functional.softmax(ema_outputs, dim=1)
        ema_pred_class = torch.argmax(ema_probs, dim=1).item()
    
    return {
        'raw_outputs': raw_outputs if isinstance(raw_outputs, np.ndarray) else raw_outputs.detach().cpu().numpy(),
        'raw_probs': raw_probs if isinstance(raw_probs, np.ndarray) else raw_probs.detach().cpu().numpy(), 
        'raw_pred_class': raw_pred_class,
        'ema_outputs': ema_outputs if isinstance(ema_outputs, np.ndarray) else ema_outputs.detach().cpu().numpy(),
        'ema_probs': ema_probs if isinstance(ema_probs, np.ndarray) else ema_probs.detach().cpu().numpy(),
        'ema_pred_class': ema_pred_class
    }

def load_and_predict(image_path, raw_model, ema_model, class_name=None):
    """Load an image from path and make predictions."""
    print(f"\n{'='*20} Predictions for {os.path.basename(image_path)} {'='*20}")
    if class_name:
        print(f"True class: {class_name}")
    
    image_data = np.load(image_path)
    
    if len(image_data.shape) == 3:
        image_data = np.expand_dims(image_data, axis=0) 
        image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)
    elif len(image_data.shape) == 4:
        image_tensor = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)
    else:
        image_tensor = torch.tensor(image_data, dtype=torch.float32) 
    
    print(f"Image tensor shape: {image_tensor.shape}")
    
    predictions = predict_single_image(raw_model, ema_model, image_tensor)
    print_predictions(predictions)
    
    return predictions

def make_predictions_from_loader(loader, raw_model, ema_model):
    """
    Makes predictions with both raw and ema model from a provided loader. Will print out a summary as
    well as save the information to both a CSV and JSON file at the end of each prediction. Makes
    predictions for each individual with a batch size of 1.
    
    Checks if cuda is available but defaults to cpu.

    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    raw_model.eval()
    ema_model.eval()

    raw_model.to(device) 
    ema_model.to(device)      

    for study_idx, (bag_imgs, _, _) in enumerate(loader):
        bag_imgs = bag_imgs.to(device)

        study_path = loader.dataset.study_dirs[study_idx] 
        study_id   = os.path.basename(study_path.rstrip(os.sep))            

        with torch.no_grad():
            raw_logits, _ = raw_model(bag_imgs)
            ema_logits, _ = ema_model(bag_imgs)

        with torch.no_grad():
            raw_outputs, raw_attentions = raw_model(bag_imgs)
            raw_probs = torch.nn.functional.softmax(raw_outputs, dim=1)
            raw_pred_class = torch.argmax(raw_probs, dim=1).item()
            
            ema_outputs, ema_attentions = ema_model(bag_imgs)
            ema_probs = torch.nn.functional.softmax(ema_outputs, dim=1)
            ema_pred_class = torch.argmax(ema_probs, dim=1).item()

        raw_probs = torch.softmax(raw_logits, dim=1)[0].cpu().numpy()
        ema_probs = torch.softmax(ema_logits, dim=1)[0].cpu().numpy()

        predictions = {
            'raw_outputs': raw_outputs if isinstance(raw_outputs, np.ndarray) else raw_outputs.detach().cpu().numpy(),
            'raw_probs': raw_probs if isinstance(raw_probs, np.ndarray) else raw_probs.detach().cpu().numpy(), 
            'raw_pred_class': raw_pred_class,
            'ema_outputs': ema_outputs if isinstance(ema_outputs, np.ndarray) else ema_outputs.detach().cpu().numpy(),
            'ema_probs': ema_probs if isinstance(ema_probs, np.ndarray) else ema_probs.detach().cpu().numpy(),
            'ema_pred_class': ema_pred_class
        }

        print_predictions(predictions, study_id=study_id)
        save_prediction(study_id, predictions)  

def print_loader_summary(loader: DataLoader,
                        max_items: int = 5,
                        indent: int = 2) -> None:
    """
    Prints a summary of tiff/dicom loader and prints out the shape of the tensors the loader
    outputs for each of the first 5 studies
    """
    pad = ' ' * indent
    dataset = loader.dataset
    n_studies = len(dataset)
    if n_studies > 1:
        print(f"\nLoader summary ➜ {n_studies} studies\n"
            f"{'-'*40}")
    else:
        print(f"\nLoader summary ➜ {n_studies} study\n"
            f"{'-'*40}")

    n_frames_all = []
    for i, (bag, label, lengths) in enumerate(loader):
        if max_items and i >= max_items:
            break

        n_frames = bag.shape[0]
        n_frames_all.append(n_frames)

        print(f"{pad}study {i:>3}: "
              f"tensor shape {tuple(bag.shape)}")
    print('-'*40 + '\n')

def main():
    parser = argparse.ArgumentParser(description='SAMIL Model Prediction')
    parser.add_argument('--study_dir', type=str, 
                        help='Top-level directory that holds study folders full of .tif/.tiff files or .dcm/.dicom files')
    args = parser.parse_args()
    study_dir = args.study_dir

    if not study_dir: 
        study_dir = choose_directory_via_gui() 
        print(study_dir)
        if not study_dir: 
            print("No directory selected - running on samples.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ema_checkpoint_path = os.path.join(script_dir, 'weights', 'best_predictions_at_ema_val', 'best_model.pth.tar')
    
    print("Loading model checkpoint...")
    raw_model, ema_model = load_model_checkpoint(ema_checkpoint_path)

    if study_dir:
        loader = build_study_loader(study_dir, batch_size=1, num_workers=2)
        print_loader_summary(loader)
        make_predictions_from_loader(loader, raw_model, ema_model)
    else:
        for class_idx in range(3):
            sample_file = os.path.join("./sample_data", f'class_{class_idx}_example.npy')
            if os.path.exists(sample_file):
                load_and_predict(sample_file, raw_model, ema_model)
            else:
                print(f"Warning: Sample file {sample_file} not found.")

if __name__ == "__main__":
    main()