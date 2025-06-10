import argparse
import os
import glob
import cv2
import numpy as np
import torch
import tqdm

from evaluation.metrics import compute_ssim_score, compute_fid_score


def load_and_prepare_image(image_path, target_size=(192, 256)):
    """Loads, resizes, and converts image to RGB numpy array."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    if img.shape[:2] != target_size[::-1]:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure RGB format
    return img


def main(args):
    # Directories for generated and ground truth images
    generated_dir = args.generated_dir
    ground_truth_dir = args.ground_truth_dir

    if not os.path.isdir(generated_dir):
        print(f"Generated directory not found: {generated_dir}")
        return
    if not os.path.isdir(ground_truth_dir):
        print(f"Ground truth directory not found: {ground_truth_dir}")
        return

    # List ground truth image files
    # Assuming ground truth filenames match the generated ones (e.g., person.jpg)
    gt_image_files = [
        f
        for f in os.listdir(ground_truth_dir)
        if os.path.isfile(os.path.join(ground_truth_dir, f))
    ]

    print(f"Found {len(gt_image_files)} ground truth images.")

    ssim_scores = []
    gen_image_tensors_for_fid = []
    gt_image_tensors_for_fid = []

    print("Computing SSIM for image pairs and preparing data for FID...")

    # Iterate through ground truth images to find corresponding generated ones
    for gt_filename in tqdm.tqdm(gt_image_files):
        gt_path = os.path.join(ground_truth_dir, gt_filename)
        # Assumes generated file has the same name as the ground truth file
        gen_path = os.path.join(
            generated_dir, gt_filename
        )  # e.g., results/test/try-on/person.jpg

        if os.path.isfile(gen_path):
            # Compute SSIM for the pair
            ssim = compute_ssim_score(gen_path, gt_path)
            if ssim is not None:
                ssim_scores.append(ssim)

            # Load and prepare tensors for FID (uint8, NCHW, RGB)
            gen_img_np = load_and_prepare_image(gen_path)
            gt_img_np = load_and_prepare_image(gt_path)

            if gen_img_np is not None and gt_img_np is not None:
                # Convert HWC RGB uint8 numpy to NCHW RGB uint8 tensor
                gen_tensor = (
                    torch.from_numpy(gen_img_np)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(torch.uint8)
                )
                gt_tensor = (
                    torch.from_numpy(gt_img_np)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(torch.uint8)
                )
                gen_image_tensors_for_fid.append(gen_tensor)
                gt_image_tensors_for_fid.append(gt_tensor)

        else:
            print(
                f"Warning: Generated image not found for {gt_filename} at {gen_path}. Skipping pair."
            )

    # Report average SSIM
    if ssim_scores:
        avg_ssim = np.mean(ssim_scores)
        print(f"\nAverage SSIM: {avg_ssim:.4f}")
    else:
        print("\nNo valid pairs found to compute SSIM.")

    # Compute FID
    print("\nComputing FID...")
    if gen_image_tensors_for_fid and gt_image_tensors_for_fid:
        # Concatenate tensors for FID computation
        # FID metric expects NCHW uint8 tensors
        try:
            fid_score = compute_fid_score(
                gen_image_tensors_for_fid, gt_image_tensors_for_fid
            )
            if fid_score is not None:
                print(f"FID Score: {fid_score:.4f}")
        except Exception as e:
            print(f"Failed to compute FID: {e}")

    else:
        print("Not enough data points to compute FID.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VTON model performance.")
    parser.add_argument(
        "--generated_dir",
        type=str,
        required=True,
        help="Directory containing generated try-on images (e.g., results/test/try-on/).",
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        required=True,
        help="Directory containing ground truth images (real photos of the person wearing the clothes).",
    )
    args = parser.parse_args()
    main(args)
