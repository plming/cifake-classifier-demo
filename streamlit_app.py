import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


# session state 초기화
if "model" not in st.session_state:
    st.session_state.model = load_model('model.h5')

st.title(":desktop_computer: AI 생성 이미지 분류기")
st.write("이 앱은 이미지가 인공지능에 의해 생성되었는지 판별하는 서비스입니다.")
st.write("모델은 *[CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)* 를 사용하여 학습되었습니다.")
st.divider()

st.header(":envelope: 이미지 업로드")
image_file = st.file_uploader(
  label="확인할 이미지를 업로드하세요.",
  type=["png", "jpg"],
  accept_multiple_files=False,
  key="image_uploader"
)
st.divider()

if image_file is not None:
    raw_image = Image.open(image_file)
    rgb_image = raw_image.convert('RGB')
    mini_image = rgb_image.resize(size=(32, 32))

    image = np.array(mini_image, dtype='float')
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = st.session_state.model.predict(image)

    is_generated_by_ai = prediction < 0.5
    if is_generated_by_ai:
        st.header(":robot_face: AI 생성 이미지입니다")
    else:
        st.header(":male-artist: AI 생성 이미지가 아닙니다")

    st.image(image_file)
    st.divider()

st.header(":page_with_curl: 같이 보기")
st.markdown("* :memo: [Google Colaboratory](https://colab.research.google.com/drive/1PL2vC3NOWrJgX7ghpu_MAxy9186owuSK?usp=sharing)")
st.markdown("* :computer: [GitHub 코드 저장소](https://github.com/plming/cifake-classifier-demo)")
st.markdown("* :chart_with_upwards_trend: [Kaggle/CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)")
