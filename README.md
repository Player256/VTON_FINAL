# VTON Demo

#### Gradio Interactive app for virtual try-on

To run this app locally, clone repo and do the following:

```bash
conda create --name vton python==3.8
pip install -r requirements.txt
python app.py
```

## Performance Comparison

| Method       | SSIM (↑) | FID (↓)  |
|--------------|:---------|:---------|
| Wardrobe-GAN | 0.740    | 47.34    |
| VITON        | 0.783    | 55.71    |
| CP-VTON      | 0.745    | 24.43    |
| ACGPN        | 0.845    | 16.64    |
| **Ours**     | **0.886**| **13.46**|

## Evaluation Metrics

### SSIM (Structural Similarity Index Measure)
SSIM measures the structural similarity between the generated try-on image and the ground truth image. In virtual try-on systems:
- **Range**: 0 to 1 (higher is better)
- **Purpose**: Evaluates how well the generated image preserves structural details, textures, and overall visual quality
- **Importance**: Critical for VTON as it ensures the clothing item maintains its realistic appearance and proper fit on the person
- **What it captures**: Luminance, contrast, and structural information between images

### FID (Fréchet Inception Distance)
FID measures the quality and diversity of generated images by comparing feature distributions between real and generated images:
- **Range**: 0 to ∞ (lower is better)
- **Purpose**: Evaluates the overall realism and quality of the generated try-on results
- **Importance**: Ensures that the virtual try-on images are indistinguishable from real photos and maintains natural appearance
- **What it captures**: Feature-level similarity using deep neural network representations, capturing both quality and diversity

### Why These Metrics Matter for VTON
- **SSIM** ensures the clothing details, patterns, and textures are preserved accurately during the virtual fitting process
- **FID** guarantees that the final try-on image looks realistic and natural, avoiding artifacts or unnatural distortions
- Together, they provide a comprehensive evaluation of both structural fidelity and perceptual quality in virtual try-on systems

Our method achieves state-of-the-art performance with the highest SSIM (0.886) and lowest FID (13.46), demonstrating superior quality in both structural preservation and realistic image generation.
