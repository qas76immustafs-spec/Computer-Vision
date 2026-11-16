# Computer Vision Lab 08 — Image Classification, Camera Calibration & Video Analytics

**Course:** Computer Vision Lab, BS(AI) 7th Semester — Bahria University Islamabad
**Student:** Qasim Mustafa (01-136232-079), BS(AI)-7A
**Notebook:** `Lab08_CV_Qasim_Mustafa.ipynb`

This notebook covers three classical computer-vision pipelines built with OpenCV, scikit-learn and scikit-image:

1. **Image classification:** HSV colour histograms and HOG features, classified with an SVM and a Random Forest
2. **Camera calibration:** finding the intrinsic matrix and lens distortion, undistorting images, and rectifying a plane with a homography
3. **Video analytics:** MOG2 motion detection, morphological cleanup, contour detection and centroid tracking with motion trails

---

## How to run

### Google Colab (recommended)
1. Open [Google Colab](https://colab.research.google.com) and choose **File → Upload notebook**, then select `Lab08_CV_Qasim_Mustafa.ipynb`.
2. Choose **Runtime → Run all**.
3. The last cell downloads `lab08_outputs.zip`, which contains every figure and the output video.

You don't need to upload any data. The notebook downloads all the datasets itself, and a full run takes about 3–5 minutes.

### Local (Jupyter / VS Code)
```bash
pip install opencv-python numpy matplotlib scikit-learn scikit-image jupyter
jupyter notebook Lab08_CV_Qasim_Mustafa.ipynb
```
The video cell uses `ffmpeg` to convert the output to H.264. Install `ffmpeg` and make sure it is on your PATH. To use a live webcam in Task 3, set `VIDEO_SOURCE = 0`. This doesn't work in Colab, because Colab runs on a remote server that can't reach your camera.

---

## Datasets

| Task | Dataset | Source |
|---|---|---|
| 1 | TensorFlow *flower_photos*: 4 classes (daisy, dandelion, roses, sunflowers), 100 images each | `storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz` |
| 2 | 13 checkerboard images, `left01–left14.jpg` (9 × 6 inner corners, 640 × 480) | OpenCV repo `samples/data` |
| 3 | `vtest.avi`: fixed surveillance camera with pedestrians (768 × 576, 10 fps, 795 frames) | OpenCV repo `samples/data` |

To classify your own images in Task 1, set `CUSTOM_DATASET_DIR` to a folder that has one sub-folder per class.

---

## Task 1: Image dataset pipeline and classical classification

**Pipeline:**
1. Load 100 images per class and resize them to 128 × 128.
2. Split the data 80/20 into training and test sets, keeping the same class balance in both (320 train / 80 test).
3. Extract features:
   - **HSV histogram:** 8 × 8 × 8 bins, giving a 512-D vector (L1-normalised)
   - **HOG:** 9 orientations, 16 × 16 cells, 2 × 2 blocks, L2-Hys normalisation, giving a 1764-D vector
   - **HSV + HOG:** both vectors joined together (2276-D)
4. Train two classifiers on each feature set:
   - **SVM:** RBF kernel, C = 10, with a `StandardScaler`
   - **Random Forest:** 300 trees
5. Evaluate with accuracy, a confusion matrix (counts and row-normalised) and a classification report.

**Results:** fill this table in from your Colab run.

| Features | SVM (RBF) | Random Forest |
|---|---|---|
| HSV | __ % | __ % |
| HOG | __ % | __ % |
| HSV + HOG | __ % | __ % |

---

## Task 2: Camera calibration, distortion correction and homography

**Pipeline:**
1. Find the board corners with `cv2.findChessboardCorners`, then refine them to sub-pixel accuracy with `cv2.cornerSubPix`.
2. Estimate the camera matrix **K** and distortion coefficients **D** with `cv2.calibrateCamera`, and calculate the reprojection error for each image and on average.
3. Plot the lens distortion as an arrow field showing how far each part of the image moves.
4. Compute a new camera matrix with `cv2.getOptimalNewCameraMatrix` (alpha = 1), undistort the image with `cv2.undistort`, and also crop it to the valid region.
5. Pick the four outer corners of the tilted board, build a homography with `cv2.getPerspectiveTransform`, and warp the board to a top-down view with `cv2.warpPerspective`.

**Results:**
```
K = [[536.07,   0.00, 342.37],
     [  0.00, 536.02, 235.54],
     [  0.00,   0.00,   1.00]]

D = [k1, k2, p1, p2, k3] = [-0.2651, -0.0467, 0.0018, -0.0003, 0.2523]

RMS reprojection error  : 0.409 px
Mean reprojection error : 0.235 px
```
A negative k1 means **barrel distortion**: straight lines bow outward, and the effect grows towards the edges of the image.

---

## Task 3: Motion detection and object tracking

**Pipeline (for each frame):**
1. **Background subtraction:** `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)`
2. **Shadow removal:** threshold the mask at 200, which drops the shadow label (127)
3. **Noise cleanup:** opening with a 3 × 3 kernel, closing with an 11 × 11 kernel, then dilation
4. **Detection:** find the outer contours, ignore any smaller than 500 px², and draw a bounding box and centre point for the rest
5. **Tracking:** a nearest-neighbour centroid tracker gives each object an ID and keeps its last 30 positions as a motion trail
6. **Overlay:** object count, FPS and frame number, written to video with `cv2.VideoWriter`

**Results:**
- 795 frames processed
- 4.66 moving objects per frame on average (maximum 8)
- 56 unique track IDs assigned

---

## Output files (`outputs/`)

| File | Description |
|---|---|
| `task1_samples.png` | Sample images from each class |
| `task1_features.png` | HOG visualisation and hue/saturation histograms |
| `task1_accuracy.png` | Accuracy for each feature set and classifier |
| `task1_confusion_matrix.png` | Confusion matrices for the best model |
| `task1_predictions.png` | Test predictions (green = correct, red = wrong) |
| `task2_corners.png` | Detected checkerboard corners |
| `task2_subpixel_zoom.png` | Raw vs. sub-pixel refined corners, zoomed in |
| `task2_reprojection_error.png` | Reprojection error for each image |
| `task2_distortion_field.png` | Lens distortion arrow field |
| `task2_undistortion.png` | Original vs. undistorted vs. cropped image |
| `task2_homography.png` | Selected region and the rectified top-down view |
| `task3_pipeline.png` | Pipeline stages for one frame, plus the object count over time |
| `task3_frames.png` | Annotated output frames |
| `task3_tracking_output.mp4` | Final tracked video (H.264) |

---

## Requirements
Python 3.x, OpenCV, NumPy, Matplotlib, scikit-learn, scikit-image, ffmpeg (already installed on Colab)
