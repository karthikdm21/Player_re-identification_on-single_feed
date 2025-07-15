# 🧠 Player Re-Identification System using YOLOv8

This project is a **real-time Player Re-Identification system** built using YOLOv8, OpenCV, and histogram-based feature matching. It tracks people (players) across video frames and assigns consistent IDs—even if they temporarily leave the frame.

---

## 🚀 Features

- YOLOv8-based person detection (Ultralytics)
- HSV color histogram for appearance features
- Spatial consistency check to re-identify across exits/entries
- Track stabilization using configurable frame counts
- Hungarian Algorithm for optimal matching
- JSON results + output video with visualized IDs

---

## 📁 Project Structure
player_reid/
├── main.py # Entry point to run the system
├── reid_system.py # Main logic for detection, tracking, ID assignment
├── track.py # Track class to manage individual player states
├── utils.py # Utility functions (IOU, histogram, similarity)
├── config.py # Configurable constants (thresholds, weights)
├── requirements.txt # Python dependencies
├── input_video.mp4 # [Your video here]
└── output_reid.mp4 # [Auto-generated output]





## ⚙️ Setup Instructions

Follow these steps to set up and run the system in your local machine.

### 1️⃣ Clone the Repository

2. For environment creation
python -m venv venv
venv\Scripts\activate

3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt



 5.  Running the Code
Place your input video inside the project directory and rename it to input_video.mp4 (or change it in main.py).

Run the following command: python main.py



4.  The program will:

Detect and track players in the video

Display bounding boxes with unique IDs

Generate output_reid.mp4 and reid_results.json after completion

Press Q to exit early during video playback.



💡 Notes
Default model used: yolov8n.pt (automatically downloaded on first run).

If you face issues with video playback, try running with show_display=False in main.py.

Works on both CPU and GPU (GPU recommended for performance).


🤝 Contributing
Pull requests are welcome! If you’d like to suggest improvements or new features, feel free to open an issue.







