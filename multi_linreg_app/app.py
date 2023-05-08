import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def main():

    st.title("Multivariate Linear Regression")
    df = pd.read_csv("https://frenzy86.s3.eu-west-2.amazonaws.com/python/data/Startup.csv")

    st.dataframe(df)

    X = df.drop(columns='Profit')
    y = df['Profit']

    test_split_size = st.slider('Split index',0.1,1.0,0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_size, random_state=667)

    
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)


    l = y_test_pred.shape[0]
    x = np.linspace(0,l,l)

    fig, ax = plt.subplots()
    ax.plot(x, y_test_pred, label='y predicted')
    ax.plot(x, y_test, label='y target')
    ax.set_title('y prediction - y target in test set')
    ax.legend()
    ax.set_xlabel('Sample index')
    ax.set_ylabel('y value')
    st.pyplot(fig)

    #inference
    st.title("Try the model")
    input1 = st.number_input("Enter a value for R&D Spend", value=0.00)
    input2 = st.number_input("Enter a float value Administration", value=0.00)
    input3 = st.number_input("Enter a float value Marketing Spend", value=0.00)
    final_input = np.array([input1, input2, input3])
    final_input = final_input.reshape(-1,3)
    pred = model.predict(final_input)
    pred_str = "%.2f" % pred[0]
    st.write("Prediction: ", pred_str, font_size="52px")
        

if __name__ == '__main__':
    main()


# streamlit run app.py


