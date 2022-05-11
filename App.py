import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression


logo, name = st.columns([1, 8])
logo.image("./img/logo.png")
name.title("Heart disease")
st.image('./img/heart1.jpg')
st.markdown("According to the CDC, heart disease is one of the leading causes of death for people of most races in the US (African Americans, American Indians and Alaska Natives, and white people). About half of all Americans (47%) have at least 1 of 3 key risk factors for heart disease: high blood pressure, high cholesterol, and smoking. Other key indicator include diabetic status, obesity (high BMI), not getting enough physical activity or drinking too much alcohol. Detecting and preventing the factors that have the greatest impact on heart disease is very important in healthcare. Computational developments, in turn, allow the application of machine learning methods to detect patterns from the data that can predict a patient's condition.")


def main():
    Races = ["American Indian/Alaskan Native",
             "Asian", "Black", "Hispanic", "Other", "White"]
    Se = ["Male", "Female"]
    Ages = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49",
            "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"]
    Diabetics = ["No", "No, borderline diabetes",
                 "Yes", "Yes (during pregnancy)"]
    GenHealths = ["Poor", "Fair", "Good", "Very good", "Excellent"]
    choices = ["No", "Yes"]

    lastname = st.sidebar.text_input("Your last name")
    firstname = st.sidebar.text_input("Your first name")
    sex = st.sidebar.radio("Sex", Se)
    Race = st.sidebar.selectbox("Race", Races)
    AgeCategory = st.sidebar.selectbox("Your age", Ages)
    Diabetic = st.sidebar.selectbox("do you have diabetic", Diabetics)
    GenHealth = st.sidebar.selectbox(
        "How can you define your general health ?", GenHealths)
    Smoking = st.sidebar.radio("Are you smoking ?", choices)
    AlcoholDrinking = st.sidebar.radio("Are you drinking alcohol ?", choices)
    Stroke = st.sidebar.radio("Did you have a stroke ?", choices)
    DiffWalking = st.sidebar.radio(
        "found you a difficulty when you walking", choices)
    PhysicalActivity = st.sidebar.radio(
        "Have you played any sports in the last month ?", choices)
    Asthma = st.sidebar.radio("do you have Asthma", choices)
    KidneyDisease = st.sidebar.radio("have you a Kidney disease", choices)
    SkinCancer = st.sidebar.radio("have you a skin cancer", choices)
    Bmi = st.sidebar.text_input("your Body Mass Index on pounds")
    PhysicalHealth = st.sidebar.slider(
        "For how many days during the past 30 days was your physical health not good ?", min_value=0, max_value=30)
    mentalHealth = st.sidebar.slider(
        "For how many days during the past 30 days was your mental health not good ?", min_value=0, max_value=30)
    SleepTime = st.sidebar.slider(
        "How many hours on average do you sleep ?", min_value=0, max_value=24)
    #model = fct.loadModel("./LogisticRegression.pkl")

    if st.sidebar.button("Done"):
        st.markdown(f"You name : **{firstname} {lastname}** <br> "
                    f"Your age between :  **{AgeCategory}** <br> "
                    f"Sex :  **{sex}** <br>"
                    f"Race :  **{Race}** <br>"
                    f"Smoking : **{Smoking}** <br>"
                    f"Drinkin alcohol :  **{AlcoholDrinking}** <br>"
                    f"Diabetic :  **{Diabetic}** <br>"
                    f"General health :  **{GenHealth}** <br>"
                    f"Stroke :  **{Stroke}** <br>"
                    f"Difficult walking : **{DiffWalking}** <br>"
                    f"Physical activity : **{PhysicalActivity}** <br>"
                    f"Asthma : **{Asthma}** <br>"
                    f"Kindy disease :  **{KidneyDisease}** <br>"
                    f"Skin cancer : **{SkinCancer}** <br>", True)

    def loadModel(filename):
        return pickle.load(open(filename, 'rb'))

    def DataConvert(var, array):
        for i in range(len(array)):
            if array[i] == var:
                return i

    def gettingDataFrame():
        sex_int = DataConvert(sex, Se)
        Race_int = DataConvert(Race, Races)
        AgeCategory_int = DataConvert(AgeCategory, Ages) + 1
        Diabetic_int = DataConvert(Diabetic, Diabetics)
        GenHealth_int = DataConvert(GenHealth, GenHealths) + 1
        Smoking_int = DataConvert(Smoking, choices)
        AlcoholDrinking_int = DataConvert(AlcoholDrinking, choices)
        Stroke_int = DataConvert(Stroke, choices)
        DiffWalking_int = DataConvert(DiffWalking, choices)
        PhysicalActivity_int = DataConvert(PhysicalActivity, choices)
        Asthma_int = DataConvert(Asthma, choices)
        KidneyDisease_int = DataConvert(KidneyDisease, choices)
        SkinCancer_int = DataConvert(SkinCancer, choices)
        features = pd.DataFrame({
            "BMICategory": [Bmi],
            "Smoking": [Smoking_int],
            "AlcoholDrinking": [AlcoholDrinking_int],
            "Stroke": [Stroke_int],
            "PhysicalHealth": [PhysicalHealth],
            "MentalHealth": [mentalHealth],
            "DiffWalking": [DiffWalking_int],
            "Sex": [sex_int],
            "AgeCategory": [AgeCategory_int],
            "Race": [Race_int],
            "Diabetic": [Diabetic_int],
            "PhysicalActivity": [PhysicalActivity_int],
            "GenHealth": [GenHealth_int],
            "SleepTime": [SleepTime],
            "Asthma": [Asthma_int],
            "KidneyDisease": [KidneyDisease_int],
            "SkinCancer": [SkinCancer_int]
        })
        return features

    features = gettingDataFrame().to_csv("./data.csv")
    df = pd.read_csv("./data.csv")
    model = loadModel("./ModelSelection/RandomForestClassifier.pkl")
    if st.button("predict"):
        prediction = model.predict(df)
        prediction_prob = model.predict_proba(df)
        if prediction == 0:
            st.markdown(f"** Hello Mr. {firstname} {lastname} The probability that you'll have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f" You are healthy!**")

        else:
            st.markdown(f"**Hello Mr. {firstname} {lastname} The probability that you will have"
                        f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                        f"  It sounds like you are not healthy.**")


if __name__ == "__main__":
    main()
