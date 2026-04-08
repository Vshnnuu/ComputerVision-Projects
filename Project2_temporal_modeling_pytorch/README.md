# Temporal Facial Behavior Detection (Project 2)
This project builds on my previous work where I extracted facial signals like smile intensity, mouth openness, and head movement from video using MediaPipe.

In this second phase, instead of relying on simple threshold-based rules, I use a **PyTorch LSTM model to learn temporal patterns** in these signals and classify facial behavior more robustly. 

The goal is to make facial behavior detection more stable and context-aware over time.

## Why Project 2?
In Project 1, everything was based on thresholds per frame. That worked, but it was very sensitive to noise.

Here, the model looks at a sequence of frames instead of a single frame. This makes predictions:

- more stable  
- less noisy  
- context-aware  
(I am using the same video as project 1 in this project, and I will soon upload a new video for the better understanding of differences of using a `pytorch` LSTM model)
For example, a brief spike in mouth opening won’t immediately trigger a "mouth_open" label unless it persists over time like it used to in Project 1 with defined thresholds.


## What this project does
with a given video of a person’s face, the system:

1. Extracts facial landmarks using MediaPipe
2. Computes three normalized signals:
   - Smile ratio
   - Mouth open ratio
   - Head turn ratio
3. Applies smoothing and preprocessing (same as Project 1)
4. Feeds sequence of 20 frames into an LSTM model
5. Predicts one of the following classes:
   - neutral
   - smiling
   - mouth_open
   - head_left
   - head_right
6. Outputs- annotated video with:
   - prediction labels
   - measurement values
   - visual landmark overlays

## Data used
The training data for this project comes from the signal extraction pipeline created in Project 1 (`signals.csv`).

### Input source
A face video containing:
- neutral expression
- smiling
- mouth opening
- head turn to the left
- head turn to the right

## Outputs

The `outputs/` folder contains the files generated after training and inference:

- `annotated_lstm_output_v2.mp4`  (Click on **"view raw"** when you open this video, and it will download locally in your computer)

- `best_lstm_classifier.pt`  
  Best model checkpoint saved during training based on validation performance.

- `lstm_classifier.pt`  
  Final saved model weights.

- `confusion_matrix.png`  
  Confusion matrix image showing class-wise prediction results.

- `loss_curve.png`  
  Training and validation loss plot across epochs.


