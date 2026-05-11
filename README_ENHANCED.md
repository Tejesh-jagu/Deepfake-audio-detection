
# Enhanced Deepfake Voice Detection System

This directory contains the enhanced modular codebase for the Deepfake Voice Detection System using Feature Fusion (Mel+MFCC), BiLSTM, and Attention Mechanism.

## Structure

- `src/config.py`: Configuration settings (Paths, Audio params, Hyperparameters).
- `src/features.py`: Feature extraction logic (MFCC, Log-Mel) and Data Augmentations (Noise, Pitch, Time, SpecAugment).
- `src/data.py`: Keras Data Generator for efficient batch processing.
- `src/model.py`: Keras Model definition (Dual-Branch CNN, BiLSTM, Attention).
- `src/evaluate.py`: Evaluation utilities (EER Calculation, ROC Plotting).
- `src/train.py`: Main training script.

## key Improvements

1.  **Feature Fusion**: Parallel processing of Log-Mel Spectrograms and MFCCs.
2.  **Attention**: Temporal attention layer to focus on spoofing cues.
3.  **Augmentation**: Robust pipeline including SpecAugment and audio signal perturbations.
4.  **Evaluation**: Implemented Equal Error Rate (EER) and AUC metrics.

## How to Run

1.  **Update Configuration**:
    Open `src/config.py` and ensure `TRAIN_DIR` points to your dataset location.

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Model**:
    ```bash
    python -m src.train
    ```
    This will:
    - Load data.
    - Train the model with early stopping.
    - Save the best model as `best_model_fusion.h5`.
    - Generate `training_history.png` and `roc_curve.png`.
    - Print test Accuracy, AUC, and EER.

## Compatibility

The code is designed for TensorFlow/Keras and assumes a dataset structure of:
- `Data-Train/for-original/training/real/*.wav` (or mp3/ogg)
- `Data-Train/for-original/training/fake/*.wav`
