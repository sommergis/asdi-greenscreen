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

@st.cache
def read_geojson(path):
    return gpd.read_file(path)


if __name__ == "__main__":
    
    #st.set_page_config(layout="wide")

    years = [2017,2018,2019,2020,2021]
    city = "Freising"
    composite_date = "2020"

    geojson_file_path = f"/home/jsommer/Schreibtisch/Ehrenamt/Freising_Urbane_Landnutzung/asdi-greenscreen/data/osm_nominatim_{city}.geojson"
    gdf = read_geojson(geojson_file_path)

    #Add sidebar to the app
    st.sidebar.markdown("Mapping and monitoring urban green areas.")
    st.sidebar.markdown("Methods and outcomes of this project shall be transferable to other cities. This is ensured by a high level of automation of the data processing and the usage of global avaliable datasets.")

    #Add title and subtitle to the main interface of the app
    col1, col2 = st.columns( [0.2, 0.8], gap="large")
    with col1:
        st.image("/home/jsommer/Schreibtisch/Ehrenamt/Freising_Urbane_Landnutzung/asdi-greenscreen/doc/logo.png", width=150)
    with col2:
        st.title("GreenScreen")
        st.subheader("Mapping and monitoring urban green areas.")

    year = st.sidebar.slider(
        'Select a year',
        min_value=min(years), max_value=max(years), value=2017, step=1
    )

    eMap = st.container() #st.expander(label="Show map")
    # eChart = st.expander(label="Show charts")
        
    # #Create three columns/filters
    #col1, col2 = st.columns([3,1])

    # with col1:
    #     # period_list=df_final["period_begin"].unique().tolist()
    #     # period_list.sort(reverse=True)
    #     # year_month = st.selectbox("Snapshot Month", period_list, index=0)
    #     pass

    # with col2:
    #     # prop_type = st.selectbox(
    #     #             "View by Property Type", ['All Residential', 'Single Family Residential', 'Townhouse','Condo/Co-op','Single Units Only','Multi-Family (2-4 Unit)'] , index=0)
    #     pass

    # with col3:
    #     pass
    #     #metrics = st.selectbox("Select Housing Metrics", ["median_sale_price","median_sale_price_yoy", "homes_sold"], index=0)

    with eMap:

        col1, col2, col3 = st.columns([0.2, 0.2, 0.6], gap="small")

        composite_date = year
        #for y in range(0, len(years)):
            #col = cols[y]
            #with col:

        with col1:
            # extent figures
            # years = [2017, 2018, 2019, 2020, 2021]
            # veg = [13443.67, 13544.49, 13436.39, 13257.82,13482.3]
            # bua = [1381.45, 1283.69, 1391.16, 1555.79,1341.69]
            # other = [130.48, 127.42, 128.05, 141.99, 131.61]

            # city core clipped figures
            # years = [2017, 2018, 2019, 2020, 2021]
            # veg = [1997.0, 2073.14, 2016.26, 1987.5, 2069.02]
            # bua = [536.78, 462.74, 518.37, 543.96, 465.83]
            # other = [20.42, 18.32, 19.57, 22.74, 19.35]

            # city osm nominatim figures
            years = [2017, 2018, 2019, 2020, 2021]
            veg = [7894.62, 7974.25, 7890.18, 7822.76, 7936.14]
            bua = [873.5, 796.52, 880.3, 940.56, 834.36]
            other = [96.65, 94.0, 94.29, 101.45, 94.27]

            years = [2017, 2018, 2019, 2020, 2021]
            veg = [7844.94, 7916.04, 7839.0, 7753.46, 7873.21]
            bua = [916.3, 848.2, 924.22, 1003.05, 888.07]
            other = [103.53, 100.53, 101.55, 108.26, 103.49]

            # years = [2017, 2018, 2019, 2020, 2021]
            # veg = [7833.63, 7907.09, 7830.83, 7746.91, 7861.85]
            # bua = [936.06, 866.29, 942.96, 1018.63, 910.32]
            # other =[6185.91, 6182.22, 6181.81, 6190.06, 6183.43]

            df = pd.DataFrame(
                columns=["year", "vegetation", "built up area", "other"],
            )
            df["vegetation"] = veg
            df["built up area"] = bua
            df["other"] = other
            df["year"] = years

            #df = df.set_index("year")

            # transform wide mode to long mode
            source = df.melt('year', var_name='category', value_name='hectares')
            #source = source.set_index("year")

            source = source[source["year"] == year]

            #st.write(source)

            colors = ['#ff0000','#0000ff', '#00ff00']

            chart = alt.Chart(source).mark_bar().encode(
                x='year:O',
                y='hectares:Q',
                color='category:N',
                #column='category:N'
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
            st.metric(label="built up area [ha]", value=source[source["category"] == "built up area"]["hectares"], delta=None, delta_color="normal", help="Built up area in hectares")
            st.metric(label="vegetation [ha]", value=source[source["category"] == "vegetation"]["hectares"], delta=None, delta_color="normal", help="Vegetation area in hectares")

        with col3:
            st.image(
                f"/home/jsommer/Schreibtisch/Ehrenamt/Freising_Urbane_Landnutzung/data/classified/esa_landcover_model_MLPClassifier_2022-08-23 11:44:18.611769_score_97_3classes/{composite_date}_osm_nominatim_{city}_clipped.png",
                #width=580,
                use_column_width="auto"
            )
            # dst_crs = "EPSG:4326"
            # with rio.open(f"/home/jsommer/Schreibtisch/Ehrenamt/Freising_Urbane_Landnutzung/data/classified/esa_landcover_model_MLPClassifier_2022-08-23 11:44:18.611769_score_97_3classes/{composite_date}_osm_nominatim_{city}_clipped.png") as src:
            #     band = src.read()
            #     geotransform = src.profile.get("transform")
            #     min_lon, min_lat, max_lon, max_lat = src.bounds
            #     print(min_lon, min_lat, max_lon, max_lat)
            #     src_crs = src.crs['init'].upper()
            
            # print(src_crs)
            # print(dst_crs)

            # # ## Conversion from UTM to WGS84 CRS
            # # [[lat_min, lon_min], [lat_max, lon_max]]
            # bounds_orig = [[min_lat, min_lon], [max_lat, max_lon]]
            # #bounds_orig = [[min_lat, max_lat], [min_lon, max_lon]]

            # bounds_fin = bounds_orig
            # bounds_fin = []
            
            # for item in bounds_orig:   
            #     #converting to lat/lon
            #     lat = item[0]
            #     lon = item[1]
                
            #     proj = Transformer.from_crs(int(src_crs.split(":")[1]), int(dst_crs.split(":")[1]), always_xy=True)

            #     lon_n, lat_n = proj.transform(lon, lat)
                
            #     bounds_fin.append([lat_n, lon_n])

            # # Finding the centre latitude & longitude    
            # centre_lon = bounds_fin[0][1] + (bounds_fin[1][1] - bounds_fin[0][1])/2
            # centre_lat = bounds_fin[0][0] + (bounds_fin[1][0] - bounds_fin[0][0])/2
            
            # # # RGB channels
            # # # bua: red
            # # r = np.where(band == 3, 255, 0).squeeze()
            # # # veg: green
            # # g = np.where(band == 1, 255, 0).squeeze()
            # # # other: blue
            # # b = np.where(band == 99, 255, 0).squeeze()

            # # im = np.stack([r,g,b], axis=0)
            # # im = reshape_as_image(im)

            # # st.image(im, width=420)
            # # #st.markdown(composite_date)

            # #
            # # add geotif overlay
            # #

            # #print(centerx, centery)
            # m = folium.Map(
            #     location=[centre_lat, centre_lon], #48.3855, 11.74],
            #     zoom_start=12,
            # )
            # basemaps = {
            #     "Google Maps": 
            #     folium.TileLayer(
            #         tiles = "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
            #         attr = "Google",
            #         name = "Google Maps",
            #         overlay = True,
            #         control = True
            #     ),
            #     "Google Satellite": folium.TileLayer(
            #         tiles = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            #         attr = "Google",
            #         name = "Google Satellite",
            #         overlay = True,
            #         control = True
            #     ),
            #     "Google Terrain": folium.TileLayer(
            #         tiles = "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
            #         attr = "Google",
            #         name = "Google Terrain",
            #         overlay = True,
            #         control = True
            #     ),
            #     "Google Satellite Hybrid": folium.TileLayer(
            #         tiles = "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
            #         attr = "Google",
            #         name = "Google Satellite",
            #         overlay = True,
            #         control = True
            #     ),
            # }
            # basemaps["Google Satellite"].add_to(m)

            # # replace -99 vals
            # #band = np.where(band == 99, 0, band)
            # #band = np.where(band == 3, 2, band)

            # im = reshape_as_image(band).squeeze()
            
            # #cmap = lambda x: (x, 1, 0, x)
                        
            # raster_layer = folium.raster_layers.ImageOverlay(
            #     image=im,
            #     mercator_project=False,
            #     bounds=bounds_fin,
            #     origin='upper',
            #     #bounds=bounds_fin,
            #     name=year,
            #     opacity=0.7,
            #     #colormap=cmap,#R,G,B,alpha
            # )
            # print(raster_layer.get_bounds())
            # raster_layer.add_to(m)

            # folium.LayerControl().add_to(m)

            # #g = folium.GeoJson(geojson_file_path, name="geojson")
            # #g.add_to(m)

            # map = st_folium(m, width=580, height=580)

