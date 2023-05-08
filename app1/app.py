import streamlit as st
import pandas as pd
import plotly.express as px
from umap import UMAP
import seaborn as sns


def main():

    st.markdown("# IRIS DATASET")
    data = st.file_uploader("Upload a Dataset", type=["csv"])

    if data is not None:
        df = pd.read_csv(data)
        st.dataframe(df)

        st.write('Dataframe Description')
        dfdesc = df.describe(include='all').T.fillna("")
        st.write(dfdesc)


        option = st.selectbox(
                'Choose a plot',
                ('UMAPScatter', 'SNSScatter', 'SNSPairplot'))
        
        if option == 'UMAPScatter':
            #plot umap 
            features = df.iloc[:, :-2]
            umap_2d = UMAP(n_components=2, init='random', random_state=0)
            proj_2d = umap_2d.fit_transform(features)
            fig = px.scatter(proj_2d, x=0, y=1,color=df['class'],width=1000,title='UMAP iris')
            st.plotly_chart(fig, theme="streamlit")

        if option == 'SNSPairplot':

            fig = sns.pairplot(df, hue='class', height=3, aspect=1)

            st.pyplot(fig)
        
        if option == 'UMAP3DScatter':

            features = df.iloc[:, :-2]

            umap_3d = UMAP(n_components=3, init='random', random_state=0)
            proj_3d = umap_3d.fit_transform(features)
            fig_3d = px.scatter_3d(proj_3d, x=0, y=1, z=2,color=df['class'], labels={'color': 'class'})
            fig_3d.update_traces(marker_size=5)
            st.plotly_chart(fig_3d, theme='streamlit')

        st.markdown("# FINE")
        


if __name__ == '__main__':
    main()





