import streamlit as st
import pandas as pd
from app_utils.load_model import load_model
import re
from app_utils.pattern_preproc import preprocess

def main():

    model = load_model()

    #['lstat', 'rm', 'ptratio', 'indus']

    input1 = st.text_input("Text to analyze: ", value='ciao')

    if input1 is not None:
        
        docx = preprocess(input1)

        doclist = []
        doclist.append(docx)

        prediction = model.predict(doclist)

        # 1 Positive, 0 Negative

        pred = 'Positive' if prediction == 1 else 'Negative'
        st.write("Prediction: ", f'<span>{pred}</span>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()