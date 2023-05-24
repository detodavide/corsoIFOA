import streamlit as st
import os
import pandas as pd
import joblib
from io import BytesIO

from app_utils.cache_convert import convert_df
from app_utils.load_model import load_model
from app_utils.pattern_preproc import preprocess

def main():

    model = load_model()
    #inference

    st.subheader("Inference uploading a text (.txt) file")
    uploaded_file = st.file_uploader("Drag and drop a TXT file here", type=["txt"], key="fileUploader")

    if uploaded_file is not None:
            
        content = uploaded_file.read()


        docx = preprocess(content)
        st.write(docx)
        
        
        doclist = []
        doclist.append(docx)

        prediction = model.predict(doclist)

        # 1 Positive, 0 Negative

        pred = 'Positive' if prediction == 1 else 'Negative'
        st.write("Prediction: ", f'<span>{pred}</span>', unsafe_allow_html=True)
        


        # Download buttons
        # csv = convert_df(df)
        # filename = "immobili_data"

        # st.download_button(
        #     label="Download data as CSV",
        #     data=csv,
        #     file_name=f'{filename}.csv',
        #     mime='text/csv',
        # )

        # buffer = BytesIO()
        # with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        #     # Write each dataframe to a different worksheet.
        #     df.to_excel(writer, sheet_name='Sheet1', index=False)
        #     # Close the Pandas Excel writer and output the Excel file to the buffer
        #     writer.save()

        #     download2 = st.download_button(
        #         label="Download data as Excel",
        #         data=buffer,
        #         file_name=f'{filename}.xlsx',
        #         mime='application/vnd.ms-excel'
        #     )

if __name__=="__main__":
    main()