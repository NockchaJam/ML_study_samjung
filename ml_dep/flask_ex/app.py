from flask import Flask, render_template, request
import numpy as np
import pickle

# 모델 파일 로드 (pickle 사용)
model_path = 'models/iris_model_svc.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

#http://127.0.0.1:5000/
@app.route('/', methods=['GET', 'POST'])
def index():
    # 사용자가 입력한 데이터를 받아서 예측 수행
    if request.method == "POST":
        # 4개의 데이터를 받아옴
        sl = float(request.form['sl'])
        sw = float(request.form['sw'])
        pl = float(request.form['pl'])
        pw = float(request.form['pw'])

        # 데이터를 2D numpy 배열로 만들어서 예측에 사용
        input_data = np.array([[sl, sw, pl, pw]])
        prediction = model.predict(input_data)[0]

        # 예측 결과에 따른 품종 이름 및 이미지 경로 설정
        iris_species = ['setosa.jpg', 'versicolor.jpg', 'virginica.png', 'flower1.jpg']
        predicted_species = iris_species[prediction]

        img_path = f"static/{predicted_species}"  # 예: /static/setosa.jpg

        # 결과와 이미지를 index.html로 전달
        return render_template('index.html', predict=predicted_species, img_path=img_path, sl=sl , sw=sw, pl=pl, pw=pw)

    # GET 요청 시 기본 입력 폼 렌더링
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)