import folium 
from folium.features import GeoJsonPopup, GeoJsonTooltip
import streamlit as st
from streamlit_folium import st_folium, folium_static
import geopandas as gpd
from folium.plugins import Draw
import pandas as pd
import altair as alt
import rasterio as rio
from rasterio.plot import reshape_as_image
from rasterio.features import dataset_features
import numpy as np
from pyproj import Transformer 
import os

@st.cache
def read_geojson(path):
    return gpd.read_file(path)


if __name__ == "__main__":

    st.set_page_config(layout="wide")

    years = [2017,2018,2019,2020,2021]
    city = "Freising"
    composite_date = "2020"

    #geojson_file_path = f"./data/data/osm_nominatim_{city}.geojson"
    #gdf = read_geojson(geojson_file_path)

    #Add sidebar to the app
    st.sidebar.markdown("## About")
    st.sidebar.markdown("GreenScreen is about mapping and monitoring urban green areas.")
    #st.sidebar.markdown("Mapping and monitoring urban green areas.")
    st.sidebar.markdown("Methods and outcomes of this project shall be transferable to other cities. This is ensured by a high level of automation of the data processing and the usage of global avaliable datasets.")

    #Add title and subtitle to the main interface of the app
    col1, col2 = st.columns( [0.2, 0.8], gap="large")
    with col1:
        st.image("../doc/logo.png", width=150)
    with col2:
        st.title("GreenScreen")
        st.subheader("Mapping and monitoring urban green areas.")

    
    year = st.sidebar.slider(
        "Year selection",
        min_value=min(years), max_value=max(years), value=2017, step=1
    )

    # tabs
    tab1, tab2 = st.tabs(["Urban area mapping", "Changes"])
    
    #
    # data
    #
    # city osm nominatim figures
    years = [2017, 2018, 2019, 2020, 2021]
    veg = [7894.62, 7974.25, 7890.18, 7822.76, 7936.14]
    bua = [873.5, 796.52, 880.3, 940.56, 834.36]
    other = [96.65, 94.0, 94.29, 101.45, 94.27]

    years = [2017, 2018, 2019, 2020, 2021]
    veg = [7844.94, 7916.04, 7839.0, 7753.46, 7873.21]
    bua = [916.3, 848.2, 924.22, 1003.05, 888.07]
    other = [103.53, 100.53, 101.55, 108.26, 103.49]

    # prepare dataframe
    df = pd.DataFrame(
        columns=["year", "vegetation", "built up area", "other"],
    )
    df["vegetation"] = veg
    df["built up area"] = bua
    df["other"] = other
    df["year"] = years

    # transform dataframe wide mode to long mode
    df_long = df.melt('year', var_name='category', value_name='hectares')

    with tab1:
        cTab1 = st.container()
        with cTab1:

            col1, col2, col3 = st.columns([0.2, 0.2, 0.6], gap="small")

            composite_date = year
        
            with col1:

                chart_source = df_long[df_long["year"] == year]

                # R, B, G                
                #colors = ['#ff0000','#0000ff', '#00ff00']
                
                # Colorbrewer orange/brown, white, teal
                #colors  = ['#d8b365','#f5f5f5','#5ab4ac']

                # logo colors: orange, white, green
                #colors = ["#f99d26", "#f5f5f5", "#56c02b"]

                colors = ["#ff8800", "#f5f5f5", "#49b675"]
                colors = ["#f99d26", "#f5f5f5", "#49b675"]

                chart = alt.Chart(chart_source).mark_bar().encode(
                    x='year:O',
                    y='hectares:Q',
                    color='category:N',
                ).properties(
                    width=200,
                    height=510
                ).configure_range(
                    category=alt.RangeScheme(colors)
                )
                st.altair_chart(chart)

            with col2:
                df_year = df_long[df_long["year"] == year]

                st.metric(label="city", value=city, help="City and year for analysis")
                st.metric(label="year", value=year, help="City and year for analysis")
                st.metric(
                    label="built up area [ha]", 
                    value=df_year[df_year["category"] == "built up area"]["hectares"],
                    delta_color="normal", 
                    help="Built up area in hectares"
                )
                st.metric(
                    label="vegetation [ha]", 
                    value=df_year[df_year["category"] == "vegetation"]["hectares"], 
                    delta_color="normal", 
                    help="Vegetation area in hectares"
                )
            with col3:
                st.image(
                    f"../data/classified/{composite_date}_osm_nominatim_{city}_clipped.png",
                    width=580,
                    use_column_width="auto"
                )

    with tab2:
        cTab2 = st.container()

        with cTab2:

            if year < 2018:
                st.subheader("No changes detectable")
                st.markdown("""Change data is visible for 2018 and beyond.
                               Please select an appropriate year in the sidebar.""")
            else:

                col1, col2, col3 = st.columns([0.2, 0.2, 0.6], gap="small")

                # 
                # Change data
                #
                if year > 2017:
                    prev_year = year - 1
                else:
                    prev_year = year

                df_year = df_long[df_long["year"] == year]
                df_prev_year = df_long[df_long["year"] == prev_year]

                bua_year = df_year[df_year["category"] == "built up area"]["hectares"].sum()
                bua_prev_year = df_prev_year[df_prev_year["category"] == "built up area"]["hectares"].sum()
                
                veg_year = df_year[df_year["category"] == "vegetation"]["hectares"].sum()
                veg_prev_year = df_prev_year[df_prev_year["category"] == "vegetation"]["hectares"].sum()

                oth_year = df_year[df_year["category"] == "other"]["hectares"].sum()
                oth_prev_year = df_prev_year[df_prev_year["category"] == "other"]["hectares"].sum()

                delta_bua = bua_year - bua_prev_year
                delta_veg = veg_year - veg_prev_year
                #no_change = oth_year - oth_prev_year

                print(f"bua: {delta_bua}, veg: {delta_veg}") #, no: {no_change}")


                with col1:
                    #
                    # Show changes in chart
                    #
                    df_change = pd.DataFrame(
                        columns=["delta_veg", "delta_bua", "no_change"],
                    )
                    df_change["current year"] = [year]
                    df_change["delta veg"] = [delta_veg]
                    df_change["delta bua"] = [delta_bua]
                    #df_change["no change"] = [no_change]

                    # # transform dataframe wide mode to long mode
                    df_change = df_change.melt("current year", var_name='category', value_name='hectares')

                    # change colors: brown, lightyellow, green
                    colors = ["#663301", "#659c3c", "#fefee0"]

                    # -2 => 101, 156, 60 (green)
                    # 0 => 254, 254, 224 (yellow)
                    # 2 => 102, 51, 1 (brown)

                    chart = alt.Chart(df_change).mark_bar().encode(
                        x='current_year:O',
                        y='hectares:Q',
                        color='category:N',
                    ).properties(
                        width=200,
                        height=510
                    ).configure_range(
                        category=alt.RangeScheme(colors)
                    )
                    st.altair_chart(chart)
                
                with col2:

                    st.metric(label="city", value=city, help="City and year for analysis")
                    st.metric(label="year", value=year, help="City and year for analysis")
                    st.metric(
                        label="built up area [ha]", 
                        value=bua_year,
                        delta=f"{delta_bua:.2f}", 
                        delta_color="normal", 
                        help="Built up area in hectares compared to previous year"
                    )
                    st.metric(
                        label="vegetation [ha]", 
                        value=veg_year, 
                        delta=f"{delta_veg:.2f}", 
                        delta_color="normal", 
                        help="Vegetation area in hectares compared to previous year"
                    )

                with col3:
                    st.image(
                        f"../data/classified/{composite_date}_osm_nominatim_{city}_clipped_{year}_diff_{prev_year}.png",
                        width=580,
                        use_column_width="auto"
                    )
