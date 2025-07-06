import streamlit as st
from PIL import Image
from ONNX0630 import predict_smiles  # 你要封装这个函数

import os
import urllib.request

model_url = "https://huggingface.co/spaces/wuzxmu/I2Mdemo/resolve/main/I2M_R4.onnx"
model_path = "./I2M_R4.onnx"

if not os.path.exists(model_path):
    urllib.request.urlretrieve(model_url, model_path)


st.set_page_config(page_title="Mol2SMILES Demo", layout="centered")

st.title("🧪 Molecular Image → SMILES Recognition")
st.write("上传化学结构图像，模型将预测 SMILES 表达式。")

uploaded_file = st.file_uploader("上传分子图像", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="你上传的图片", use_column_width=True)

    with st.spinner("模型正在识别，请稍候..."):
        smiles = predict_smiles(image)
        st.success(f"识别结果 SMILES: `{smiles}`")



