from flask import Flask,render_template
import numpy as np
import pickle

# 모델 파일 로드 (pickle 사용)
model_path = 'models/iris_model_svc.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

#http://127.0.0.1:5500/
@app.route('/' , methods = ['GET', 'POST'])
def index():
    aaa  = "헬로 플라스크"
    bbb = "static/setosa.jpg"
    print(aaa)
    # return render_template('index.html')
    return render_template('index.html', predict=aaa, img_path = bbb)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port="5000", debug=True)