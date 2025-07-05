import gradio as gr
from ONNX0630 import predict_from_image  # 你需要在 ONNX0630.py 中定义这个函数

demo = gr.Interface(
    fn=predict_from_image,
    inputs=gr.Image(type="pil", label="Upload Chemical Structure Image"),
    outputs=gr.Textbox(label="Predicted SMILES"),
    title="Structure → SMILES Recognizer",
    description="Upload an image of a chemical structure, and get the predicted SMILES string."
)

demo.launch()
