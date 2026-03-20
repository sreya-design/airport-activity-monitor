from ultralytics import YOLO

model = YOLO("yolo26n.pt")  # downloads automatically on first run

model.train(
    data="data/data.yaml",
    epochs=20,
    imgsz=320,
    batch=4,
    device="cpu",
    workers=2,
    name="aircraft_cpu"
)

print("Done! Model saved to: runs/detect/aircraft_cpu/weights/best.pt")