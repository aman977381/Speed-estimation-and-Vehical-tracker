## Vehicels Speed Estimation with YOLO 🚗

In this demonstration, we apply YOLOv8 object detection models and ByteTrack, an efficient online multi-object tracking technique, to estimate vehicle speeds. The process involves using the supervision package for tasks such as tracking and annotations.


https://github.com/aman977381/Speed-estimation-and-Vehical-tracker/assets/50145701/b17e9d86-c189-419b-aa73-ed520e7ef714



> [!IMPORTANT]
> Make sure to modify the [`SOURCE`](https://github.com/aman977381/Speed-estimation-and-Vehicel-tracker/blob/d2b496ec0efd23c314ab1c59054ff978fb02910a/app.py#L9)
> and [`TARGET`](https://github.com/aman977381/Speed-estimation-and-Vehicel-tracker/blob/d2b496ec0efd23c314ab1c59054ff978fb02910a/app.py#L13)
> settings based on your specific camera setup when implementing the speed estimation code on your footage.

# Setup

- Clone the repository and enter the speed estimation example directory:

  ```bash
  git clone https://github.com/aman977381/Speed-estimation-and-Vehicel-tracker.git
  cd Speed-estimation-and-Vehicel-tracker
  ```
- Install necessary libraries:
  ```bash
  pip install -r requirements.txt
  ```
- Acquire the vehicles.mp4 video file:
  ```bash
  python3.10 video_downloader.py
  ```

## 🛠️ script arguments
  - `--source_video_path`: Required. The path to the source video file that will be
  analyzed. This is the input video on which traffic flow analysis will be performed.
## Run
  ```bash
    python ultralytics_example.py \
    --source_video_path data/vehicles.mp4 
  ```
