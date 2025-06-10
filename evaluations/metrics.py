import torch
import torchmetrics
import numpy as np
import cv2


def compute_ssim_score(img1_path, img2_path):
    """
    Computes SSIM between two images given their paths.
    Images are loaded, resized to 256x192, converted to grayscale (SSIM typically grayscale),
    and SSIM is calculated.

    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image (ground truth).

    Returns:
        float: SSIM score between the two images, or None if error.
    """
    try:
        # Load and resize images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"Error loading images: {img1_path} or {img2_path}")
            return None

        # Ensure images are the same size (the VTON model outputs 256x192)
        target_size = (192, 256)  # (width, height)
        if img1.shape[:2] != target_size[::-1]:
            img1 = cv2.resize(img1, target_size, interpolation=cv2.INTER_AREA)
        if img2.shape[:2] != target_size[::-1]:
            img2 = cv2.resize(img2, target_size, interpolation=cv2.INTER_AREA)

        # Convert to grayscale (SSIM is often computed on luminance)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Convert to torch tensors (NCHW format, float [0, 1])
        # SSIM usually works on grayscale 0-1 or 0-255, let's stick to 0-255 for torchmetrics
        # torchmetrics SSIM expects uint8 (H, W) or (N, C, H, W)
        img1_tensor = (
            torch.from_numpy(img1_gray).unsqueeze(0).unsqueeze(0).to(torch.uint8)
        )
        img2_tensor = (
            torch.from_numpy(img2_gray).unsqueeze(0).unsqueeze(0).to(torch.uint8)
        )

        # Compute SSIM
        ssim_metric = torchmetrics.StructuralSimilarityIndexMeasure(data_range=255)
        ssim_score = ssim_metric(img1_tensor, img2_tensor)

        return ssim_score.item()

    except Exception as e:
        print(f"Error computing SSIM for {img1_path} and {img2_path}: {e}")
        return None


def compute_fid_score(imgs1_tensors, imgs2_tensors):
    """
    Computes FID between two lists of image tensors.
    Tensors are expected to be uint8 and NCHW format, with pixel values in [0, 255].

    Args:
        imgs1_tensors (list[torch.Tensor]): List of tensors for the first set of images (generated).
        imgs2_tensors (list[torch.Tensor]): List of tensors for the second set of images (ground truth).

    Returns:
        float: FID score, or None if error.
    """
    if not imgs1_tensors or not imgs2_tensors:
        print("FID computation requires non-empty image lists.")
        return None

    # Torchmetrics FID expects uint8 NCHW tensors with pixel values in [0, 255]
    # Ensure all tensors are NCHW and uint8 [0, 255]
    def prepare_tensors(tensors_list):
        prepared = []
        for t in tensors_list:
            # Assuming tensors are already CHW or NCHW float/uint8 from loader
            if t.dim() == 3:  # CHW
                t = t.unsqueeze(0)  # NCHW
            if t.dtype != torch.uint8:
                if torch.max(t) <= 1.0:  # Assuming 0-1 float range
                    t = (t * 255).to(torch.uint8)
                else:  # Assuming >1 float range (maybe 0-255 float?)
                    t = t.to(torch.uint8)
            prepared.append(t)
        return torch.cat(prepared, dim=0)

    imgs1_tensor = prepare_tensors(imgs1_tensors)
    imgs2_tensor = prepare_tensors(imgs2_tensors)

    try:
        # Compute FID
        fid_metric = torchmetrics.FID().to(
            imgs1_tensor.device
        )  # Move metric to device where tensors are
        fid_metric.update(imgs1_tensor, real=False)
        fid_metric.update(imgs2_tensor, real=True)
        fid_score = fid_metric.compute()

        return fid_score.item()

    except Exception as e:
        print(f"Error computing FID: {e}")
        return None
