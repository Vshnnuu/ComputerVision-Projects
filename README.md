# Computer Vision Projects – Facial Behavior Analysis

## Overview
The work is divided into two stages:
- **Project 1** – Signal extraction and rule-based detection  
- **Project 2** – Temporal modeling using PyTorch (LSTM)

Both projects use the same core idea: extracting meaningful signals from facial landmarks using MediaPipe. The difference lies in how those signals are interpreted.

## Project 1 – Facial Signal Extraction and Rule-Based Detection
In Project 1, I built a pipeline to extract facial behavior signals directly from video using MediaPipe Face Mesh.

### What it does
- Detects facial landmarks for each frame
- Computes normalized measurements:
  - Smile ratio (mouth width / face width)
  - Mouth open ratio (lip gap / face height)
  - Head turn ratio (nose position relative to face center)
- Applies smoothing to reduce noise
- Uses threshold-based rules to classify behavior per frame

### Output
- Annotated video showing:
  - landmark points
  - measurement lines
  - computed ratios
  - detected labels (rule-based)

### Key characteristics
- Frame-by-frame decision making  
- No learning involved  
- Fully deterministic logic  
- Works well for clear, exaggerated expressions  
- Sensitive to small fluctuations and noise
  

## Project 2 – Temporal Modeling with PyTorch (LSTM)
In Project 2, I extend the same signal pipeline but replace rule-based decisions with a learned model.

Instead of treating each frame independently, the model looks at a sequence of frames and learns patterns over time.

### What it does

- Uses the same signals generated in Project 1
- Converts them into sequences (window of 20 frames)
- Trains an LSTM model using PyTorch
- Predicts behavior based on temporal context

### Classes predicted

- neutral  
- smiling  
- mouth_open  
- head_left  
- head_right  

### Output

- Annotated video similar to Project 1, but:
  - labels are predicted by the trained model
  - predictions are more stable over time

### Key characteristics

- Sequence-based prediction  
- Uses temporal context  
- More robust to noise and small fluctuations  
- Learns patterns instead of relying on fixed thresholds  

---

## Key Difference Between the Two Projects

The main difference is not in how the signals are extracted, but in how they are interpreted.

| Aspect | Project 1 | Project 2 |
|------|--------|--------|
| Approach | Rule-based | Learned (PyTorch) |
| Input | Single frame | Sequence of frames |
| Logic | Thresholds | LSTM model |
| Stability | Can be noisy | More stable |
| Adaptability | Fixed rules | Learns patterns |

### This transition reflects a common pattern in computer vision systems:
1. Start with interpretable signal extraction  
2. Move towards learning-based models  
3. Incorporate temporal context 

Project 2 builds directly on Project 1 and I used the same Video in both of them to analyse and evaluate, although I will soon record a new video with similar expressions and movements but enough to differentiate between both approaches.
---

