from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(
    data=("data.yaml"),
    epochs=30,
    imgsz=640,
    patience=3
)