import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('animal.h5')

st.title("AI생성 이미지 분류기")
st.write("이 앱은 진짜와 AI가 생성한 이미지를 분류해줍니다.")

st.subheader("체험해보기")
upload = st.file_uploader(
  '확인할 이미지를 업로드하세요',
  type=['png','jpg'],
  accept_multiple_files=False
)

print("fooo")
# display uploaded raw image only
if upload is not None:
    CLASS_NAME = ["AI 생성 이미지", "진짜 이미지"]
    image = Image.open(upload)
    # run keras model
    image = image.resize((32, 32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    image = image.astype('float32')
    image = image / 255.0
    pred = model.predict(image)[0]
    idx = int(pred > 0.5)
    st.write(pred)
    st.write("이 이미지는", CLASS_NAME[idx], "입니다.")
    st.image(upload, caption='Uploaded Image.', use_column_width=True)

st.subheader("참고자료")
st.markdown("* [Kaggle/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)")