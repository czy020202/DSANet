# DSANet: 3D Deformable Slice-Aware Network with Adaptive Slice Grouping for Robust Pulmonary Nodule Detection
🎉 Our paper has been accepted by ECAI2025.
## Requirements
The code is built with the following libraries:
- Python 3.8
- PyTorch1.12.1
- CUDA 11.6
- pydicom
- scikit-image
- termcolor
- pynrrd
- tqdm
- tensorboard
- nibabel

Besides, you need to install a custom module for bounding box NMS and overlap calculation.
```
cd build/box
python setup.py install
```