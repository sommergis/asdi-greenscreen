import folium 
import streamlit as st
import pandas as pd
import altair as alt
import rasterio as rio
import os
import time

if __name__ == "__main__":

    st.set_page_config(layout="wide")

    years = [2017,2018,2019,2020,2021]
    city = "Freising"
    composite_date = "2020"

    #
    # sidebar
    #
    # Add sidebar to the app
    st.sidebar.write("# About")
    st.sidebar.write(
        """
        **GreenScreen** is about mapping and monitoring urban green areas as a contribution to the [Amazon Sustainability Data Initiative (ASDI) Global Hackathon 2022](https://aws-asdi.devpost.com/?ref_feature=challenge&ref_medium=discover).
        """
    )
    sidebar_exp = st.sidebar.expander(label="More info", expanded=True)
    with sidebar_exp:
        st.write(
            """
            >Urban green areas are important for a healthy city climate. 
            >Parks, green spaces and gardens improve the air quality and the urban climate, they dampen noise,
            >are habitats for animals and plants and thus contribute to the
            >protection of species and the preservation of biodiversity.
            *[Source: BMI](https://www.bmi.bund.de/SharedDocs/downloads/DE/publikationen/themen/bauen/wohnen/gruenbuch-stadtgruen.pdf?__blob=publicationFile&v=3)*
            """
        )
        st.write(
            """**GreenScreen** wants to contribute to number 11 of the UN's Sustainability Development Goals [Sustainable Cities and Communities](https://en.wikipedia.org/wiki/Sustainable_Development_Goal_11) 
            by analyzing the development of urban green areas by the example of [Freising](https://www.openstreetmap.org/search?query=Freising#map=12/48.3899/11.7165) a small city near Munich, Germany.
            """
        )

    sidebar = st.sidebar
    with sidebar:
        city = st.selectbox(label="Select a city", options=("Freising", "Agadir"))

        state = st.session_state

        state.year = 2017

        # TODO: does not work with slider
        #st.sidebar.button("Play", on_click=increment_year())

        state.year = st.slider(
            "Year selection",
            min_value=min(years), 
            max_value=max(years),
            value=state.year,
            step=1
        )

        # UN logo
        st.image("../doc/SDG-Icons-2019_WEB/E-WEB-Goal-11.png", width=150)

    #
    # Add title and subtitle to the main interface of the app
    #
    col1, col2 = st.columns( [0.2, 0.8], gap="large")
    with col1:
        st.image("../doc/logo.png", width=150)
    with col2:
        st.title("GreenScreen")
        st.subheader("Mapping and monitoring urban green areas.")

    #
    # tabs
    #
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Urban area mapping", "Changes", "Data & Methods", "Processing Workflows", "Tools"])

    #
    # data
    #
    years = [2017, 2018, 2019, 2020, 2021, 2017, 2018, 2019, 2020, 2021]
    veg = [7844.94, 7916.04, 7839.0, 7753.46, 7873.21,6466.39, 6197.19, 6187.74, 5250.4, 6008.25]
    bua = [916.3, 848.2, 924.22, 1003.05, 888.07,5048.02, 5291.09, 5316.92, 6254.81, 5493.5]
    other = [103.53, 100.53, 101.55, 108.26, 103.49,31.06, 57.19, 40.81, 40.26, 43.72]

    # prepare dataframe
    df = pd.DataFrame(
        columns=["city", "year", "vegetation", "non-vegetation area", "other"],
    )
    df["city"] = ["Freising","Freising","Freising","Freising","Freising","Agadir","Agadir","Agadir","Agadir","Agadir"]
    df["vegetation"] = veg
    df["non-vegetation area"] = bua
    df["other"] = other
    df["year"] = years

    # transform dataframe wide mode to long mode
    df_long = df.melt(['city', 'year'], var_name="category", value_name='hectares')

    with tab1:
        cTab1 = st.container()
        with cTab1:

            col1, col2, col3 = st.columns([0.2, 0.2, 0.6], gap="small")

            composite_date = state.year
        
            with col1:

                chart_source = df_long[(df_long["year"] == state.year) & (df_long["city"] == city)]

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
                df_year = df_long[(df_long["year"] == state.year) & (df_long["city"] == city)]

                st.metric(label="city", value=city, help="City and year for analysis")
                st.metric(label="year", value=state.year, help="City and year for analysis")
                st.metric(
                    label="non-vegetation area [ha]", 
                    value=df_year[df_year["category"] == "non-vegetation area"]["hectares"],
                    delta_color="normal", 
                    help="non-vegetation area in hectares"
                )
                st.metric(
                    label="vegetation [ha]", 
                    value=df_year[df_year["category"] == "vegetation"]["hectares"], 
                    delta_color="normal", 
                    help="Vegetation area in hectares"
                )
            with col3:
                st.image(
                    f"../data/classified/{city}_{composite_date}_clipped.png",
                    width=580,
                    use_column_width="auto"
                )

    with tab2:
        cTab2 = st.container()

        with cTab2:

            if state.year < 2018:
                st.subheader("No changes detectable")
                st.markdown("""Change data is visible for 2018 and beyond.
                               Please select an appropriate year in the sidebar.""")
            else:

                col1, col2, col3 = st.columns([0.2, 0.2, 0.6], gap="small")

                # 
                # Change data
                #
                if state.year > 2017:
                    prev_year = state.year - 1
                else:
                    prev_year = state.year

                df_year = df_long[(df_long["year"] == state.year) & (df_long["city"] == city)]
                df_prev_year = df_long[(df_long["year"] == prev_year) & (df_long["city"] == city)]

                bua_year = df_year[df_year["category"] == "non-vegetation area"]["hectares"].sum()
                bua_prev_year = df_prev_year[df_prev_year["category"] == "non-vegetation area"]["hectares"].sum()
                
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
                        columns=["delta vegegation", "delta non-vegetation", "no change"],
                    )
                    df_change["current year"] = [state.year]
                    df_change["delta vegegation"] = [delta_veg]
                    df_change["delta non-vegetation"] = [delta_bua]
                    #df_change["no change"] = [no_change]

                    # # transform dataframe wide mode to long mode
                    df_change = df_change.melt("current year", var_name='category', value_name='hectares')

                    # change colors: brown, lightyellow, green
                    colors = ["#663301", "#659c3c", "#fefee0"]

                    chart = alt.Chart(df_change).mark_bar().encode(
                        x='current year:O',
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
                    st.metric(label="year", value=f"{prev_year} to {state.year}", help="City and year for analysis")
                    st.metric(
                        label="non-vegetation area [ha]", 
                        value=bua_year,
                        delta=f"{delta_bua:.2f}", 
                        delta_color="normal", 
                        help="Non vegetation area in hectares compared to previous year"
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
                        f"../data/classified/{city}_clipped_{state.year}_diff_{prev_year}.png",
                        width=580,
                        use_column_width="auto"
                    )

    with tab3:
        st.write("""
            ### Overview
            The following methods were evaluated regarding their application in the project:
            - classical remote sensing analysis of satellite data
            - machine learning / deep learning based classification
            - change detection
            - development of a processing chain and a web app

            #### Location and time period
            The city of Freising, Germany shall be processed in this project as an example. It turned out during the project that another city - Agadir, Morocco could be analyzed and thus is also presented in the web app.
            Data shall be analyzed since 2017 until 2021.

            ### Description
            #### Data preparation
            First we need to get the city boundary. While there exist some global administrative boundary geodatabases these are heterogeneous for each country in terms of details. 
            So we decided to go for the OSM Nominatim web service of OpenStreemap. Because of its service nature we don't have to download and process data first - we can just query a service for a certain geometry.
            Then we need Sentinel-2 scenes of the ESA Copernicus programme. To get all available scenes of a city we need to query the STAC Service from Element84.
            Afterwards we read all satellite scenes for each year - but only the small part of the scene with the location of the selected city. This is efficient due to the Cloud-optimized GeoTIFF format of the [ASDI's](https://nachhaltigkeit.aboutamazon.de/umwelt/die-cloud/asdi) Sentinel-2 COG archive.
            During the creation of the median composites per year cloudy pixels will be masked out for quality reasons. After that the ESA Worldcover 2020 product from [ASDI](https://nachhaltigkeit.aboutamazon.de/umwelt/die-cloud/asdi) will be clipped to the city's extent and transformed to the Sentinel scenes' spatial reference system.
            #### Machine learning model
            >The idea is to train a machine learning model on the median composite of Sentinel-2 for the year 2020 with the ESA Worldcover 2020 map as ground truth for the landuse classification. 
            Then the trained model shall predict vegetation, built up area/bare soil or other pixels for the years 2017,2018,2019,2021 and (if possible) 2022.
            Now we can prepare the input features for the training of the machine learning model: compute remote sensing indices like NDVI, NDWI, NDBI and some GLCM texture metrics such as homogeneity and entropy.
            Several classification algorithms of the scikit learn package were tested - RandomForestClassifier and MLPClassifier performed best while MLPClassifier was even slightly better on recognizing textures like bigger buildings than RandomForestClassifier.

        """)
        st.write(
            """
            #### Think global - act local
            The methods and outcomes of this project shall be transferable to other cities. 
            This is ensured by a high level of automatization of the processing chain and the usage of global avaliable datasets."""
        )
        st.write(
            """
            #### Data
            Datasets used for the project are mostly from the [Amazon Sustainability Data Initiative (ASDI)](https://nachhaltigkeit.aboutamazon.de/umwelt/die-cloud/asdi) as follows:
            - [ESA Worldcover 2020](https://registry.opendata.aws/esa-worldcover/)
            - [Sentinel-2 Cloud-Optimized GeoTIFFs](https://registry.opendata.aws/sentinel-2-l2a-cogs/)

            Additional Service used:
            - [OSM Nominatim](https://nominatim.openstreetmap.org/ui/search.html)
            """
        )
        st.write(
            """
            #### Products
            ##### **Zonal statistics for city boundary**
            Aggregated results of vegetation and non-vegetation area shall be dervied at city boundary level for each year. 

            ##### **Pixel level maps for each year**
            Furthermore various maps at pixel level analysis shall be compiled that show the urban green area development for each year.

            ##### **Change detection pixel level maps**
            Furthermore various maps at pixel level analysis shall be compiled that show the change of the urban green area development for each year.

            """
        )

    with tab4:
        st.subheader("Workflows")
        st.write(
            """#### Core processing workflow"""
        )
        st.image("../doc/workflow/GreenScreen Core Processing Workflow.png")

        st.write(
            """#### Change detection workflow"""
        )
        st.image("../doc/workflow/GreenScreen Change Detection Workflow.png")

    with tab5:
        st.write(
            """
            #### Tools
            The following tools helped us in the project
            - [Amazon SageMaker Studio Lab](https://studiolab.sagemaker.aws/)
            - [rasterio](https://github.com/rasterio/rasterio)
            - [scikit-learn](https://scikit-learn.org/)
            - [scikit-image](https://scikit-image.org/)
            - [fast-glcm](https://github.com/tzm030329/GLCM)
            - [yed Live](https://www.yworks.com/yed-live/)
            - [streamlit](https://www.streamlit.io/)        
            - [Bottleneck](https://kwgoodman.github.io/bottleneck-doc/index.html)        
            - [sat-search](https://github.com/sat-utils/sat-search)    
            - [Earth Search STAC](https://earth-search.aws.element84.com/v0)
            - [OSM Nominatim](https://nominatim.openstreetmap.org/ui/search.html)
            """
        )