# Bowling Scoreboard Data Extraction

## Overview
This repository provides an automated Computer Vision and Optical Character Recognition (OCR) pipeline to extract player names, frame-by-frame shot symbols, cumulative frame scores, and total game scores from video footage of a bowling scoreboard (`data/bowling_scoreboard.mp4`).

## Problem Statement
The goal of this project is to process bowling alley scoreboard video recordings and accurately convert the visual grid into structured digit and symbol data outputs (JSON and CSV formats) while rendering an annotated video overlay demonstrating real-time extraction tracking.

## Approach
The data extraction pipeline follows a sequential multi-stage architecture:

$$\text{Video Input} \rightarrow \text{Frame Sampling} \rightarrow \text{Scoreboard ROI Detection} \rightarrow \text{Multi-Variant Image Preprocessing} \rightarrow \text{Grid/Cell Segmentation} \rightarrow \text{Smart Whitelisted OCR} \rightarrow \text{Temporal Aggregation} \rightarrow \text{JSON / CSV / Video Output}$$

1. **Video Input & Sampling**: Loads the video and extracts frames at a configurable sampling rate (e.g. 1.0-second intervals).
2. **Scoreboard ROI Detection**: Locates the primary scoreboard display region within the video frame.
3. **Multi-Variant Image Preprocessing**: Generates adaptive contrast-enhanced, thresholded, and border-padded variants per cell to handle active (yellow background) and inactive (blue background) row highlights.
4. **Grid / Cell Segmentation**: Slices player rows into individual sub-cells for player names, shot sub-boxes, cumulative frame scores, and total scores.
5. **OCR & Domain Filtering**: Runs EasyOCR with cell-specific character whitelists (`JVPT` for names, `0123456789` for scores, `0123456789X/-` for shots) and applies bowling arithmetic rules (converting valid pin sum overflows $S_1 + S_2 \ge 10$ to spare `/`).
6. **Temporal Aggregation**: Evaluates observations across multiple sampled timestamps using majority voting and progressive non-decreasing cumulative score constraints.
7. **Structured Output Generation**: Exports data into JSON and CSV files and generates an annotated MP4 video.

## Technologies
- **Python 3.10+**
- **OpenCV (`opencv-python-headless`)**: Video I/O, image transformations, CLAHE, and thresholding.
- **EasyOCR**: Character recognition for digit and symbol extraction.
- **NumPy & Pandas**: Matrix manipulations and tabular data structuring.
- **PyTorch**: Underlying deep learning engine for EasyOCR.

## Project Structure
```
computer-vision/
├── data/
│   └── bowling_scoreboard.mp4
├── src/
│   ├── main.py
│   ├── config.py
│   ├── video_processor.py
│   ├── scoreboard_detector.py
│   ├── image_preprocessor.py
│   ├── grid_segmenter.py
│   ├── ocr_processor.py
│   ├── temporal_aggregator.py
│   ├── data_extractor.py
│   └── visualizer.py
├── output/
│   ├── extracted_scoreboard.csv
│   ├── extracted_scoreboard.json
│   └── annotated_scoreboard.mp4
├── screenshots/
│   ├── input_frame.png
│   ├── detected_scoreboard.png
│   ├── preprocessed_scoreboard.png
│   ├── segmented_grid.png
│   ├── final_output.png
│   └── ocr_cells/
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation
Ensure you have Python 3.10 or higher installed. Clone the repository and install dependencies using `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage
Run the complete extraction pipeline with the following command:

```bash
python src/main.py --video data/bowling_scoreboard.mp4
```

## Output
Running `main.py` generates the following output files in the `output/` and `screenshots/` directories:

- **`output/extracted_scoreboard.csv`**: Tabular CSV file containing rows for each player and frame with `Player`, `Frame`, `Shot`, `CumulativeScore`, and `Total` columns.
- **`output/extracted_scoreboard.json`**: Structured JSON containing complete video metadata, final extracted scoreboard state, and frame-by-frame temporal snapshot history.
- **`output/annotated_scoreboard.mp4`**: Annotated output video showing the scoreboard bounding box, grid cell boundaries, and visual text overlays.
- **`screenshots/`**: Visual verification screenshots showing detection stages and preprocessed OCR debug cell crops (`screenshots/ocr_cells/`).

## Implementation Details
- **Sub-box Slicing**: Slices individual shot sub-boxes (`shot_1`, `shot_2`, `shot_3`) independently to prevent character boundary collisions.
- **Adaptive Contrast & Inversion**: Applies CLAHE and Otsu thresholding with black/white border padding to prevent text character clipping on crop edges.
- **Domain Arithmetic Validation**: Automatically corrects 2-digit shot candidates $S_1 S_2$ where $S_1 + S_2 \ge 10$ to spare notation `S1/`.
- **Temporal Voting**: Aggregates readings across multiple video frames to eliminate transient visual occlusions and pin-reset animations.

## Limitations
- **Camera Movement / Zoom**: The grid segmenter uses fixed layout relative coordinates calibrated for standard static camera angles.
- **Severe Graphic Occlusions**: During pin reset animations when total scores temporarily flicker, extraction relies on temporal snapshot voting.
- **Unplayed Future Frames**: Frames 5–10 in the sample video are unplayed and correctly output as blank entries.

## Future Improvements
- Add dynamic keypoint/corner alignment (e.g. ArUco or homography transformation) to handle moving camera footage.
- Integrate lightweight custom CRT font classifier models for faster CPU inference speeds.
