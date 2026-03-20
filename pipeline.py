from ultralytics import YOLO
from PIL import Image, ImageDraw
from classify import classify_crop

detector = YOLO("runs/detect/aircraft_cpu/weights/best.pt")

def run_pipeline(image_path: str) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    results = detector(image_path, device="cpu")[0]
    draw = ImageDraw.Draw(img)

    if len(results.boxes) == 0:
        draw.text((10, 10), "No aircraft detected", fill="red")
        return img

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)

        # Make sure crop is valid
        if x2 <= x1 or y2 <= y1:
            continue

        crop = img.crop((x1, y1, x2, y2))
        label = classify_crop(crop)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Draw label background
        label_w = len(label) * 7
        draw.rectangle([x1, y1 - 18, x1 + label_w, y1], fill="red")
        draw.text((x1 + 2, y1 - 16), label, fill="white")

    return img

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    result = run_pipeline(path)
    result.save("output.jpg")
    print("Saved to output.jpg")
    result.show()