# cifake-classifier-demo

## 소개

이 앱은 이미지가 인공지능에 의해 생성되었는지 판별하는 서비스입니다.
모델은 [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)를 사용하여 학습되었습니다.

## 링크

https://cifake.streamlit.app

## 실행

1. 의존성 설치

```bash
uv sync
```

2. 로컬에서 앱 실행

```bash
uv run streamlit run streamlit_app.py
```

## 모델 학습 노트북

```bash
uv sync --group notebook
uv run --group notebook jupyter lab create_model.ipynb
```
