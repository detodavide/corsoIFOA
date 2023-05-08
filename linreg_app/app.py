import streamlit as st
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():

    st.title("Linear Regression")
    n_points = st.slider('Number of points', 100, 300, 100)
    generate_random = np.random.RandomState(667)
    x = 10 * generate_random.rand(n_points)

    dev_st = st.slider('Noise Range', 1,10,1)
    noise = np.random.normal(0,dev_st,n_points)
    y = 3*x + noise
    X = x.reshape(-1,1)

    test_split_size = st.slider('Split index',0.1,1.0,0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_size, random_state=667)

    
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    fig, ax = plt.subplots()
    ax.plot(y_test_pred, label='y predicted')
    ax.plot(y_test, label='y target')
    ax.set_title('y prediction - y target in test set')
    ax.legend()
    ax.set_xlabel('Sample index')
    ax.set_ylabel('y value')
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.scatter(X_train, y_train)
    ax.scatter(X_test, y_test)
    ax.plot(X_train, y_train_pred, 'r')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_title('Scatter plot with regression line')
    st.pyplot(fig)
        

if __name__ == '__main__':
    main()


# streamlit run app.py


