import streamlit as st
import pandas as pd
import plotly.express as px
from umap import UMAP
import seaborn as sns
from datetime import datetime, date
import matplotlib.pyplot as plt
from meteostat import Point, Daily
import plotly.graph_objects as go


def main():

    st.markdown("# TIME SERIES")
    start_ = st.date_input(
                            "Start date",
                            date(2019, 7, 6))

    start = datetime(start_.year, start_.month, start_.day)


    end_ = st.date_input(
                        "End date",
                        date(2023, 5, 17))
    
    end = datetime(end_.year, end_.month, end_.day)

    city_lat = st.number_input("Place latitude", format="%.6f")
    city_long = st.number_input("Place longitude", format="%.6f")

    city_name = st.text_input("Place name")

    cities = {city_name:[city_lat, city_long]}

    ###############################################################

    city = Point(list(cities.values())[0][0],list(cities.values())[0][1], 20)

        # Get daily data for 2018
    df = Daily(city, start, end)
    df = df.fetch()
    df['city'] = list(cities.keys())[0]
        
    st.dataframe(df)

    if start and end and city_lat and city_long and city_name:

        fig1 = plt.figure(figsize=(14,12))
        plt.plot(df['tavg'])
        plt.plot(df['tmin'])
        plt.plot(df['tmax'])
        st.pyplot(fig1)

        # import plotly.graph_objects as go
        fig = go.Figure()

        #Actual 
        fig.add_trace(go.Scatter(x = df.index, 
                                y = df['tavg'],
                                mode = "lines",
                                name = "Aveg",
                                line_color='#0000FF',
                                ))
        ##############################################################
        #Predicted 
        fig.add_trace(go.Scatter(x = df.index, 
                                y = df['tmax'],
                                mode = "lines", 
                                name = "Max",
                                line_color='#ff8c00',
                                ))

        ##############################################################
        # adjust layout
        fig.update_layout(title = "Titolo",
                        xaxis_title = "Date",
                        yaxis_title = "Sales",
                        width = 1700,
                        height = 700,
                        )
        ####################################################################
        # zoomming
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=2, label="3y", step="year", stepmode="backward"),
                    dict(count=10, label="10y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        fig.add_vline(x=date.today(), line_width=3, line_dash="dash", line_color="red")
        fig.update_layout(width=850)
        st.plotly_chart(fig)


if __name__ == '__main__':
    main()





