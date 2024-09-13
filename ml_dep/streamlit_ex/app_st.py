import streamlit as st
import numpy as np
from PIL import Image
import pickle

# 모델 파일 로드 (pickle 사용)
model_path = 'models/iris_model_svc.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Iris 데이터셋의 품종 이름
iris_target_names = ['setosa', 'versicolor', 'virginica']

# 페이지 제목 및 설명
st.title("Iris 품종 예측 서비스")
st.write("꽃받침과 꽃잎의 길이 및 너비를 각각 입력하세요. 모든 값을 입력한 후 예측 버튼을 클릭하세요.")

# 사용자가 직접 값을 입력하는 슬라이더 (feature 값을 조정할 수 있도록 +, - 버튼으로 수정)
sepal_length = st.number_input('Sepal Length (cm)', min_value=4.0, max_value=8.0, step=0.1, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=2.0, max_value=4.5, step=0.1, value=3.0)
petal_length = st.number_input('Petal Length (cm)', min_value=1.0, max_value=7.0, step=0.1, value=1.5)
petal_width = st.number_input('Petal Width (cm)', min_value=0.1, max_value=2.5, step=0.1, value=0.2)

# 예측 버튼
if st.button('예측하기'):
    # 입력값을 모델에 넣어 예측
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_class = iris_target_names[prediction[0]]

    # 예측 결과 출력
    st.subheader(f"예측된 품종: {predicted_class}")

    # 이미지 출력 함수
    def display_iris_image(label):
        if label == 'setosa':
            image_path = 'static/setosa.jpg'
        elif label == 'versicolor':
            image_path = 'static/versicolor.jpg'
        else:
            image_path = 'static/virginica.png'

        image = Image.open(image_path)
        st.image(image, caption=f"{label.capitalize()}", use_column_width=True)

    # 예측된 품종에 따른 이미지 출력
    display_iris_image(predicted_class)