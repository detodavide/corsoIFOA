import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import joblib
import numpy as np

def add_bg_image():
    st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url('https://w.wallhaven.cc/full/g8/wallhaven-g89dgl.png');
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
    )

def main():

    #add_bg_image()
    model = joblib.load("3input_regression_startup.pkl")
    #inference
    st.title("Try the model")

    st.subheader("Inference uploading a dataset")
    data = st.file_uploader("Upload a Dataset", type=["csv"])

    if data is not None:

        df = pd.read_csv(data)
        df = df.round()
        if df['Profit'] is not None:
            df.drop(columns='Profit', inplace=True)
        st.dataframe(df)

        st.write('Dataframe Description')
        dfdesc = df.describe(include='all').T.fillna("")
        st.write(dfdesc)

        df_pred = model.predict(df)
        df_pred = df_pred.round()
        df['Profit'] = df_pred
        st.write('Updated Dataframe')
        st.dataframe(df)


    if data is None:
        st.subheader("Inference with manual inputs")
        input1 = st.number_input("Enter a value for R&D Spend", value=0.00)
        input2 = st.number_input("Enter a float value Administration", value=0.00)
        input3 = st.number_input("Enter a float value Marketing Spend", value=0.00)
        final_input = np.array([input1, input2, input3])
        final_input = final_input.reshape(-1,3)
        pred = model.predict(final_input)
        pred_str = "%.2f" % pred[0]
        st.write("Prediction: ", pred_str)
        

if __name__ == '__main__':
    main()


# streamlit run app.py


