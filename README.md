---
title: VTON Demo
colorFrom: yellow
colorTo: red
sdk: gradio
app_file: app.py
pinned: false
---


#### Gradio Interactive app for virtual try-on

To run this app locally, clone repo and do the following:
```python
conda create --name vton python==3.8
pip install -r requirements.txt
python app.py
```

| Method       | SSIM (↑)        FID (↓)  |
|--------------|:--------------|:--------:|
| Wardrobe-GAN | 0.740           47.34    |
| VITON        | 0.783           55.71    |
| CP-VTON      | 0.745           24.43    |
| ACGPN        | 0.845           16.64    |
| Ours         | **0.886**       **13.46**|


