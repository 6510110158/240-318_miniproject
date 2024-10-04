import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# โหลดโมเดลที่บันทึกไว้
model = joblib.load('model.pkl')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # รับข้อมูลจากผู้ใช้
    data = request.get_json()
    sleep_hours = data.get('sleep_hours')
    work_hours = data.get('work_hours')
    stress_level = data.get('stress_level')
    emotion_state = data.get('emotion_state')

    # สร้าง DataFrame จากข้อมูลที่ป้อนเข้ามา
    user_data = pd.DataFrame({
        'sleep_hours': [sleep_hours],
        'work_hours': [work_hours],
        'stress_level': [stress_level],
        'emotion_state': [emotion_state]
    })

    # ทำการวินิจฉัย
    prediction = model.predict(user_data)

    # แปลงผลการวินิจฉัยเป็นข้อความ
    depression_levels = {0: "ไม่มีอาการซึมเศร้า", 1: "มีอาการซึมเศร้าเล็กน้อย", 
                         2: "มีอาการซึมเศร้าขั้นรุนแรง"}
    predicted_level = depression_levels[prediction[0]]

    return jsonify({"prediction": predicted_level})

if __name__ == '__main__':
    app.run(debug=True)
