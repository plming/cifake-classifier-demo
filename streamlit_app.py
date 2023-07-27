import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model.h5')

st.title("AI생성 이미지 분류기")
st.write("이 앱은 진짜와 AI가 생성한 이미지를 구별합니다.")

st.subheader("이미지 업로드")
upload = st.file_uploader(
  '확인할 이미지를 업로드하세요',
  type=['png','jpg'],
  accept_multiple_files=False
)

if upload is not None:
    image = Image.open(upload)
    # run keras model
    image = image.resize((32, 32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image.astype('float')
    image = image / 255.0
    print(image)
    print(image.shape)
    pred = model.predict(image)
    print(pred)
    idx = int(pred > 0.5)

    CLASS_NAME = ["AI 생성 이미지", "진짜 이미지"]
    st.write("이 이미지는", CLASS_NAME[idx], "입니다.")
    st.image(upload, caption=CLASS_NAME[idx])

st.subheader("참고자료")
st.markdown("* [Kaggle/CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)")