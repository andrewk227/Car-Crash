from ultralytics import YOLO

# Making the prediction

model = YOLO(
    "runs/detect/train4/weights/best.pt"
)  # load a pretrained model (recommended for training)


video_path = "data/testing.mp4"
no_acc = "not_acc.mp4"


resultsAccidents = model.track(source=video_path, conf=0.7, vid_stride=5, show=True)
