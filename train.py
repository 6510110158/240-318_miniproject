import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

data = {
    'sleep_hours': [
        6, 5, 7, 8, 4, 6, 7, 5, 9, 3, 
        5, 6, 8, 5, 7, 6, 6, 5, 3, 8, 
        6, 5, 4, 7, 5, 5, 9, 10, 4
    ],  # จำนวนชั่วโมงนอน
    'work_hours': [
        10, 9, 12, 8, 15, 10, 11, 9, 7, 16, 
        10, 8, 9, 10, 10, 12, 11, 7, 9, 8, 
        9, 6, 10, 10, 5, 7, 8, 9, 10
    ],  # ชั่วโมงทำงานต่อวัน
    'stress_level': [
        8, 7, 6, 4, 10, 8, 6, 7, 3, 10, 
        5, 7, 6, 5, 8, 7, 9, 6, 7, 8, 
        5, 4, 9, 10, 4, 3, 5, 7, 8
    ],  # ระดับความเครียด 0-10
    'emotion_state': [
        2, 3, 1, 4, 1, 2, 3, 3, 5, 1, 
        2, 2, 1, 2, 3, 4, 2, 3, 4, 2, 
        1, 2, 3, 3, 2, 1, 2, 3, 4
    ],  # ระดับอารมณ์ 0-5
    'depression': [
        2, 1, 0, 0, 2, 1, 0, 1, 0, 2, 
        1, 1, 2, 1, 1, 2, 0, 0, 1, 1, 
        2, 0, 1, 1, 2, 1, 1, 1, 0
    ]  # 0 = ไม่มี, 1 = มีอาการซึมเศร้าเล็กน้อย, 2 = มีอาการซึมเศร้าปานกลาง
}


# แปลงเป็น DataFrame
df = pd.DataFrame(data)

# แยก Features (X) และ Target (y)
X = df[['sleep_hours', 'work_hours', 'stress_level', 'emotion_state']]
y = df['depression']

# แบ่งข้อมูลเป็น Train และ Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สเกลข้อมูลด้วย StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# บันทึก scaler ไว้ใช้ใน predict.py
joblib.dump(scaler, 'scaler.pkl')

# สร้างโมเดล Logistic Regression โดยใช้ class_weight='balanced'
model = LogisticRegression(multi_class='ovr', solver='liblinear', class_weight='balanced')

# เทรนโมเดล
model.fit(X_train, y_train)

# บันทึกโมเดล
joblib.dump(model, 'model.pkl')
print("โมเดลและ scaler ถูกบันทึกลงใน 'model.pkl' และ 'scaler.pkl' เรียบร้อยแล้ว")
