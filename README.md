# Video Autoencoder for UCSD Anomaly Detection

This repository contains a PyTorch implementation of an autoencoder-based model for anomaly detection in video frames, specifically designed for the UCSD Anomaly Dataset (Ped1 and Ped2).

## Features
- Frame-level anomaly detection using a Residual Autoencoder architecture
- Training and evaluation scripts
- Utilities for dataset handling and ground truth conversion
- Model checkpointing

## Project Structure
```
├── train.py                # Training script
├── eval.py                 # Evaluation script
├── generate_labels.py      # Ground truth label generator
├── models/                 # Model architectures
├── utils/                  # Dataset and utility functions
├── checkpoints/            # Saved model weights (ignored by git)
├── data/                   # UCSD Anomaly Dataset (ignored by git)
```

## Dataset
- Download the UCSD Anomaly Dataset v1p2 from the official source: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
- Place the extracted folders inside `data/UCSD_Anomaly_Dataset.v1p2/`
- The dataset is **not** included in this repository and is ignored by `.gitignore`.

## Setup
1. Clone the repository:
   ```powershell
   git clone https://github.com/yourusername/video_autoencoder.git
   cd video_autoencoder
   ```
2. (Optional) Create and activate a Python virtual environment:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install torch torchvision tqdm
   ```

## Training
Run the training script:
```powershell
python train.py
```
Model checkpoints will be saved in the `checkpoints/` directory.

## Evaluation
Run the evaluation script:
```powershell
python eval.py
```

## Generating Ground Truth Labels
To generate ground truth labels for evaluation:
```powershell
python generate_labels.py
```

## Customization
- Modify model architecture in `models/aa_rae.py` and `models/residual_block.py`.
- Adjust dataset handling in `utils/dataset.py`.

## Contributing
Pull requests and issues are welcome! Please open an issue for major changes.

## License
This project is licensed under the MIT License.

## Acknowledgements
- UCSD Anomaly Dataset: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
- PyTorch: https://pytorch.org/
