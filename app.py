import streamlit as st
import pandas as pd
import numpy as np
import joblib

# تحميل النموذج والـ Scaler
@st.cache_resource
def load_model():
    model = joblib.load("model/random_forest_heart_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# عنوان التطبيق
st.title("🌡️ تطبيق التنبؤ بأمراض القلب")
st.markdown("### باستخدام نموذج Random Forest")

st.write("""
أدخل البيانات الطبية للمريض للتنبؤ بإمكانية وجود مرض قلبي.
""")

# أعمدة البيانات الأصلية (يجب أن تطابق بالضبط أسماء الأعمدة في الملف)
feature_names = [
    'age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
    'fasting blood sugar', 'resting ecg', 'max heart rate',
    'exercise angina', 'oldpeak', 'ST slope'
]

# إنشاء نموذج إدخال
st.sidebar.header("بيانات المريض")

input_data = {}
for feature in feature_names:
    if feature == 'age':
        val = st.sidebar.slider("العمر (age)", 20, 100, 50)
    elif feature == 'sex':
        val = st.sidebar.selectbox("الجنس (sex)", options=[0, 1], format_func=lambda x: "أنثى" if x == 0 else "ذكر")
    elif feature == 'chest pain type':
        val = st.sidebar.selectbox("نوع ألم الصدر", options=[1, 2, 3, 4],
                                   format_func=lambda x: {1:"نموذجي", 2:"غير نموذجي", 3:"غير قلبي", 4:"بدون أعراض"}[x])
    elif feature == 'resting bp s':
        val = st.sidebar.slider("ضغط الدم أثناء الراحة (mm Hg)", 80, 200, 120)
    elif feature == 'cholesterol':
        val = st.sidebar.slider("الكوليسترول (mg/dl)", 100, 600, 200)
    elif feature == 'fasting blood sugar':
        val = st.sidebar.selectbox("سكر الدم الصائم > 120 mg/dl", options=[0, 1], format_func=lambda x: "لا" if x == 0 else "نعم")
    elif feature == 'resting ecg':
        val = st.sidebar.selectbox("نتيجة تخطيط القلب أثناء الراحة", options=[0, 1, 2],
                                   format_func=lambda x: {0:"طبيعي", 1:"شذوذ ST-T", 2:"تضخم بطين أيسر"}[x])
    elif feature == 'max heart rate':
        val = st.sidebar.slider("أقصى معدل نبض (bpm)", 60, 220, 150)
    elif feature == 'exercise angina':
        val = st.sidebar.selectbox("ألم صدري ناتج عن التمرين", options=[0, 1], format_func=lambda x: "لا" if x == 0 else "نعم")
    elif feature == 'oldpeak':
        val = st.sidebar.slider("انخفاض ST الناتج عن التمرين (oldpeak)", 0.0, 6.2, 1.0, step=0.1)
    elif feature == 'ST slope':
        val = st.sidebar.selectbox("ميل مقطع ST", options=[1, 2, 3],
                                   format_func=lambda x: {1:"تصاعدي", 2:"مسطح", 3:"تنازلي"}[x])
    
    input_data[feature] = val

# زر التنبؤ
if st.sidebar.button("التنبؤ بحالة القلب"):
    # تحويل الإدخال إلى DataFrame
    input_df = pd.DataFrame([input_data])
    
    # تقييس البيانات بنفس الـ Scaler المستخدم في التدريب
    input_scaled = scaler.transform(input_df)
    
    # التنبؤ
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    st.markdown("## 📊 نتيجة التنبؤ")
    
    if prediction == 1:
        st.error("⚠️ **تحذير: يُحتمل وجود مرض قلبي**")
        st.write(f"احتمالية الإصابة: {prediction_proba[1]*100:.1f}%")
    else:
        st.success("✅ **لا يوجد مرض قلبي (طبيعي)**")
        st.write(f"احتمالية السلامة: {prediction_proba[0]*100:.1f}%")
    
    st.info("ملاحظة: هذا النموذج لأغراض تعليمية فقط، لا يُغني عن استشارة الطبيب.")

# عرض بعض المعلومات الإضافية
st.sidebar.markdown("---")
st.sidebar.caption("نموذج Random Forest مدرب على بيانات Cleveland + Hungary")