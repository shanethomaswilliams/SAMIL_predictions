import os, tempfile, argparse, warnings
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image, ImageSequence, ImageOps, TiffImagePlugin
import pydicom
from pydicom.encaps import generate_pixel_data_frame

try:
    import cv2
except ImportError:
    cv2 = None
    warnings.warn("OpenCV not found - MPEG-4 DICOM clips will be skipped.")

RESIZED     = 112
TIFF_MEAN   = [0.059]*3
TIFF_STD    = [0.138]*3


class TIFFDataset(Dataset):
    def __init__(self, tiff_folder_path: str, tiff_filenames: List[str]):
        self.tiff_folder_path = tiff_folder_path
        self.tiff_files = sorted(tiff_filenames)  # Use only specified files
        self.resized_shape = 112
        
        # Echo dataset normalization - same as original
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.059, 0.059, 0.059], std=[0.138, 0.138, 0.138])
        ])
        
        self.processed_images = self._process_all_tiffs()
        
    def _resize_and_pad(self, tiff_image) -> Tuple[int, int, tuple]:
        """Resize and calculate padding - exact same as original"""
        original_height = tiff_image.size[1]
        original_width = tiff_image.size[0]
        
        if original_height > original_width:
            new_height = self.resized_shape
            new_width = int(self.resized_shape * (original_width / original_height))
            pad_along_width = self.resized_shape - new_width
            pad_along_height = 0
            pad_configuration = ((0, pad_along_height), (0, pad_along_width), (0, 0))
        else:
            new_width = self.resized_shape
            new_height = int(self.resized_shape * (original_height / original_width))
            pad_along_height = self.resized_shape - new_height
            pad_along_width = 0
            pad_configuration = ((0, pad_along_height), (0, pad_along_width), (0, 0))
            
        return new_height, new_width, pad_configuration
    
    def _process_single_tiff(self, tiff_path: str) -> np.ndarray:
        """Process single TIFF - exact same pipeline as original"""
        try:
            im = Image.open(tiff_path)
            
            for page, tiff_image in enumerate(ImageSequence.Iterator(im)):
                tiff_image = ImageOps.grayscale(tiff_image)
                new_height, new_width, pad_configuration = self._resize_and_pad(tiff_image)
                tiff_image = tiff_image.resize((new_width, new_height))
                tiff_image_array = np.expand_dims(np.array(tiff_image), axis=2)
                tiff_image_array = np.pad(tiff_image_array, pad_width=pad_configuration, 
                                        mode='constant', constant_values=0)
                break  # Only first frame
            
            return tiff_image_array
            
        except Exception as e:
            print(f"Error processing {tiff_path}: {str(e)}")
            return np.zeros((self.resized_shape, self.resized_shape, 1), dtype=np.uint8)
    
    def _process_all_tiffs(self) -> np.ndarray:
        """Process only the specified TIFF files"""
        processed_images = []
        
        for tiff_filename in self.tiff_files:
            tiff_path = os.path.join(self.tiff_folder_path, tiff_filename)
            if os.path.exists(tiff_path):
                processed_frame = self._process_single_tiff(tiff_path)
                processed_images.append(processed_frame)
            else:
                print(f"Warning: TIFF file not found: {tiff_path}")
        
        if processed_images:
            return np.array(processed_images)
        else:
            return np.zeros((1, self.resized_shape, self.resized_shape, 1), dtype=np.uint8)
    
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        # Same as original function
        rgb_images = np.repeat(self.processed_images, 3, axis=3)
        
        transformed_images = []
        for img in rgb_images:
            pil_img = Image.fromarray(img.astype(np.uint8))
            transformed = self.transform(pil_img)
            transformed_images.append(transformed)
        
        bag_images = torch.stack(transformed_images)
        dummy_label = torch.tensor(-1, dtype=torch.long)
        dummy_relevance = torch.ones(len(self.tiff_files), dtype=torch.float32)
        
        return bag_images, dummy_label, dummy_relevance

class DICOMDataset(Dataset):
    def __init__(self, folder: str, files: List[str]):
        self.folder   = folder
        self.dicom_files = sorted(files)
        self.resized_shape = RESIZED
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=TIFF_MEAN, std=TIFF_STD)
        ])
        self.processed_images = self._process_all_dicoms()

    def _video_frames_from_dicom(self, ds) -> List[np.ndarray]:
        if cv2 is None:
            return []
        frames = []
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            for frag in generate_pixel_data_frame(ds.PixelData):
                tmp.write(frag)
            tmp_name = tmp.name
        cap = cv2.VideoCapture(tmp_name)
        while cap.isOpened():
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            frames.append(f)
        cap.release()
        os.remove(tmp_name)
        return frames

    def _first_frame(self, path: str) -> np.ndarray:
        try:
            ds = pydicom.dcmread(path, force=True)
            if 'MPEG' in ds.file_meta.TransferSyntaxUID.name and cv2:
                frames = self._video_frames_from_dicom(ds)
                frame  = frames[0] if frames else ds.pixel_array[0]
            else:
                px = ds.pixel_array
                frame = px[0] if px.ndim == 3 else px
            frame = _arr_to_uint8(frame)
            pil, pad = _resize_pad(Image.fromarray(frame), self.resized_shape)
            arr = np.expand_dims(np.array(pil), 2)
            arr = np.pad(arr, pad, mode='constant')
            return arr
        except Exception as e:
            print(f"DICOM error {path}: {e}")
            return np.zeros((self.resized_shape, self.resized_shape, 1), np.uint8)

    def _process_all_dicoms(self) -> np.ndarray:
        imgs = [self._first_frame(os.path.join(self.folder, f))
                for f in self.dicom_files]
        return np.asarray(imgs) if imgs else np.zeros((1,self.resized_shape,
                                                       self.resized_shape,1),np.uint8)
    def __len__(self): return 1
    def __getitem__(self, idx):
        rgb = np.repeat(self.processed_images, 3, axis=3)
        imgs = [self.transform(Image.fromarray(i)) for i in rgb]
        bag  = torch.stack(imgs)
        return bag, torch.tensor(-1), torch.ones(len(self.dicom_files))

class StudyFolderDataset(Dataset):
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)

        if _has_images(self.root_dir):
            self.study_dirs = [self.root_dir]     
        else:
            self.study_dirs = [os.path.join(self.root_dir, d)
                               for d in sorted(os.listdir(self.root_dir))
                               if os.path.isdir(os.path.join(self.root_dir, d))
                               and _has_images(os.path.join(self.root_dir, d))]
        if not self.study_dirs:
            raise RuntimeError(f'No DICOM/TIFF files found under {root_dir}')

        self.datasets = []
        for sdir in self.study_dirs:
            dicoms = [f for f in os.listdir(sdir)
                      if f.lower().endswith(('.dcm', '.dicom'))]
            tiffs  = [f for f in os.listdir(sdir)
                      if f.lower().endswith(('.tif', '.tiff'))]
            if dicoms:
                self.datasets.append(DICOMDataset(sdir, dicoms))
            elif tiffs:
                self.datasets.append(TIFFDataset(sdir, tiffs))
            else:
                raise RuntimeError(f'{sdir} contains no readable images')

    def __len__(self) -> int:
        return len(self.datasets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.datasets[idx][0] 


def _build_loader(ds: Dataset,
                  batch_size: int,
                  num_workers: int) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())

def _has_images(path: str) -> bool:
    return any(f.lower().endswith(('.dcm', '.dicom', '.tif', '.tiff'))
        for f in os.listdir(path))

def build_tiff_study_loader(study_dir: str,
                            batch_size: int = 1,
                            num_workers: int = 4) -> DataLoader:
    tiffs = [f for f in os.listdir(study_dir)
             if f.lower().endswith(('.tif', '.tiff'))]
    if not tiffs:
        raise RuntimeError(f'No TIFF files found in {study_dir}')
    ds = TIFFDataset(study_dir, tiffs)
    return _build_loader(ds, batch_size, num_workers)

def build_dicom_study_loader(study_dir: str,
                             batch_size: int = 1,
                             num_workers: int = 4) -> DataLoader:
    dicoms = [f for f in os.listdir(study_dir)
              if f.lower().endswith(('.dcm', '.dicom'))]
    if not dicoms:
        raise RuntimeError(f'No DICOM files found in {study_dir}')
    ds = DICOMDataset(study_dir, dicoms)
    return _build_loader(ds, batch_size, num_workers)

def build_study_loader(root_dir: str,
                       batch_size: int = 1,
                       num_workers: int = 4) -> DataLoader:
    ds = StudyFolderDataset(root_dir)
    return _build_loader(ds, batch_size, num_workers)

def build_dicom_as_tiff_loader(study_dir: str,
                               batch_size: int = 1,
                               num_workers: int = 4,
                               out_dir: str = None) -> DataLoader:
    n_written = convert_dicom_study_to_tiffs(study_dir, out_dir or study_dir)
    if n_written:
        print(f"Wrote {n_written} TIFFs from DICOM frames.")
    return build_tiff_study_loader(out_dir or study_dir,
                                   batch_size=batch_size,
                                   num_workers=num_workers)



def _resize_pad(pil: Image.Image,
                size: int = RESIZED) -> Tuple[np.ndarray, Tuple[int, int, Tuple]]:
    h, w = pil.height, pil.width
    if h > w:
        new_h, new_w = size, int(size * w / h)
        pad = ((0, 0), (0, size - new_w), (0, 0))
    else:
        new_w, new_h = size, int(size * h / w)
        pad = ((0, size - new_h), (0, 0), (0, 0))
    pil = pil.resize((new_w, new_h))
    return pil, pad

def _arr_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.max() > 0:
        arr = 255 * (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    return arr.round().astype(np.uint8)

def _save_tiff(img_2d: np.ndarray, path: str) -> None:
    img = Image.fromarray(img_2d, mode="L")
    img.save(path, format="TIFF", compression="none")

def convert_dicom_study_to_tiffs(study_dir: str, out_dir: str = None) -> int:
    if out_dir is None:
        out_dir = study_dir
    os.makedirs(out_dir, exist_ok=True)

    written = 0
    for dcm_name in sorted(f for f in os.listdir(study_dir)
                           if f.lower().endswith(".dcm")):
        dcm_path = os.path.join(study_dir, dcm_name)
        base_out = os.path.join(out_dir, os.path.splitext(dcm_name)[0])

        if any(fname.startswith(os.path.basename(base_out))
               for fname in os.listdir(out_dir)):
            continue

        try:
            ds = pydicom.dcmread(dcm_path, force=True)
            if 'MPEG' in ds.file_meta.TransferSyntaxUID.name and cv2:
                frames = DICOMDataset._video_frames_from_dicom(
                    DICOMDataset, ds)
                for i, f in enumerate(frames):
                    f = _arr_to_uint8(f)
                    pil, pad = _resize_pad(Image.fromarray(f), RESIZED)
                    arr = np.pad(np.expand_dims(np.array(pil), 2), pad,
                                 mode='constant')
                    _save_tiff(arr.squeeze(), f"{base_out}_frame{i:04d}.tiff")
                    written += 1
            else:
                px = ds.pixel_array
                if px.ndim == 2:
                    px = px[None, ...]
                for i, frame in enumerate(px):
                    frame = _arr_to_uint8(frame)
                    pil, pad = _resize_pad(Image.fromarray(frame), RESIZED)
                    arr = np.pad(np.expand_dims(np.array(pil), 2), pad,
                                 mode='constant')
                    _save_tiff(arr.squeeze(), f"{base_out}_frame{i:04d}.tiff")
                    written += 1
        except Exception as e:
            print(f"Convert_dicom_study_to_tiffs: {dcm_name}: {e}")
    return written
