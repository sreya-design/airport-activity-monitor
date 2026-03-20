import gradio as gr
import numpy as np
from PIL import Image
from pipeline import run_pipeline

def predict(img_array):
    if img_array is None:
        return None
    img = Image.fromarray(img_array)
    img.save("_input.jpg")
    result = run_pipeline("_input.jpg")
    return np.array(result)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload aerial or satellite image"),
    outputs=gr.Image(label="Detected aircraft"),
    title="Aircraft Detector",
    description="Upload an aerial image. Detects and classifies aircraft using YOLO26 + EfficientNet. Runs on CPU.",
    examples=[],
)

demo.launch(inbrowser=True)