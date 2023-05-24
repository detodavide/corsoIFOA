import streamlit as st

from app_pages.file_inference import main as df_inf
from app_pages.m_input_inference import main as input_inf
from app_utils.bg_image import add_bg_from_url

def main():

    st.set_option('deprecation.showPyplotGlobalUse', False)
    add_bg_from_url()

    st.title("Sentiment Analysis")

    st.subheader("Select between manual input or from a csv/xlsx file as input data")

    options = ['FILE INPUT INFERENCE', 'MANUAL INPUT INFERENCE']
    selected_option = st.selectbox('Select an option: ', options)
 

    if selected_option == "FILE INPUT INFERENCE":
        df_inf()
        
        

    if selected_option == "MANUAL INPUT INFERENCE":
        input_inf()
        

if __name__ == '__main__':
    
    main()