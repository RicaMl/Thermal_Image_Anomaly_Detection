# R&D AI – Thermal Image Anomaly Detection for Satellite Payloads

This repository contains a complete Jupyter notebook dedicated to analyzing and automatically detecting thermal anomalies on calibration images from infrared cameras aboard satellites. The tool combines computer vision techniques (OpenCV) and unsupervised machine learning (Isolation Forest) to simulate, segment, and classify suspicious regions.

## Objective

Infrared cameras on satellites regularly transmit calibration images. The goal is to develop an automated pipeline capable of:

1. **Simulating** realistic thermal images with various types of defects (photon noise, hot pixels, local overheating gradients).
2. **Extracting** regions of interest via mathematical morphology operations and computing geometric and thermal features.
3. **Automatically classifying** these regions to distinguish normal noise fluctuations from genuine critical anomalies.

This work is part of predictive maintenance and optical telemetry analysis for space applications.

## Main Features

- **Synthetic image generation** – configurable (size, noise, defective pixels, overheating gradient).
- **OpenCV pipeline** – adaptive thresholding, morphological closing, contour detection, bounding box extraction.
- **Feature extraction** – area, maximum and mean temperature, circularity.
- **Unsupervised AI model** – Isolation Forest trained on extracted regions to classify each zone as “nominal” or “critical anomaly”.
- **Visualizations**:
  - Multi‑case dashboard (raw images, masks, histograms).
  - Decision boundary and confusion matrix.
  - Final diagnosis on a test image under simulated flight conditions.

## Notebook Structure

1. **Library imports** – numpy, pandas, cv2, matplotlib, scikit‑learn, etc.
2. **Vision module** – functions for thermal image generation and region extraction.
3. **Visualization** – analysis of different study cases (normal, hot pixels, local overheating, multi‑anomalies).
4. **AI integration** – generation of a region database, training of the Isolation Forest.
5. **Evaluation** – display of the decision boundary and confusion matrix.
6. **Real‑condition test** – simulation of a noisy image with real‑time diagnosis.

## Requirements

- Python 3.8 or higher
- Libraries listed in the notebook: `numpy`, `pandas`, `opencv-python`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`

Install dependencies with:

```bash
pip install numpy pandas opencv-python matplotlib seaborn scipy scikit-learn
```

## Usage

1. Open `Detection_Anomalies_Images_Thermiques.ipynb` in Jupyter Lab / Notebook or Google Colab.
2. Run cells sequentially.
3. Intermediate and final results (graphs, metrics) are displayed directly in the notebook.

## Expected Results

- The OpenCV pipeline correctly isolates hot regions.
- The Isolation Forest achieves decent accuracy (about 83% for critical anomalies in the presented example).
- A confusion matrix and classification report allow performance evaluation.
- The final diagnosis annotates detected regions on an unseen test image.

## Visualization Preview

- **Study cases** – raw images, binary masks, temperature histograms.
- **Decision boundary** – area vs maximum temperature, colored by AI prediction.
- **Confusion matrix** – comparison with an empirical ground truth.
- **Final output** – raw thermal image and annotated image with AI verdict (red = critical, yellow = nominal).

## Context

Developed as part of an R&D project in artificial intelligence for space‑based thermal image analysis.

## License

This project is provided for educational and demonstration purposes. Generated images and data are synthetic. Free to use and adapt with attribution to the source.
