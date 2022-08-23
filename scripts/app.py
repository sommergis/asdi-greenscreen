import folium 
from folium.features import GeoJsonPopup, GeoJsonTooltip
import streamlit as st
from streamlit_folium import st_folium, folium_static
import geopandas as gpd
from folium.plugins import Draw
import pandas as pd


@st.cache
def read_geojson(path):
    return gpd.read_file(path)


if __name__ == "__main__":
    
    city = "Freising"
    composite_date = "2020"

    geojson_file_path = f"/home/jsommer/Schreibtisch/Ehrenamt/Freising_Urbane_Landnutzung/asdi-greenscreen/data/osm_nominatim_{city}.geojson"
    gdf = read_geojson(geojson_file_path)

    #Add sidebar to the app
    st.sidebar.markdown("### GreenScreen")
    st.sidebar.image("/home/jsommer/Schreibtisch/Ehrenamt/Freising_Urbane_Landnutzung/asdi-greenscreen/doc/logo.png")
    st.sidebar.markdown("Methods and outcomes of this project shall be transferable to other cities. This is ensured by a high level of automation of the data processing and the usage of global avaliable datasets.")
    #Add title and subtitle to the main interface of the app
    st.title("GreenScreen")
    st.markdown("Mapping and monitoring urban green areas.")

    
    c = st.container()
    
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

    with c:
        m = folium.Map(
            location=[48.3855, 11.74],
            tiles="cartodbpositron",
            zoom_start=12,
        )
        folium.GeoJson(geojson_file_path, name="geojson").add_to(m)
        folium.LayerControl().add_to(m)

        output = st_folium(m, width=700, height=600)

        #with col2:
        year = [2020,]
        veg = [13257.82,]
        bua = [1555.79,]
        other = [141.99, ]
        df = pd.DataFrame(
            columns=["year", "vegetation [ha]", "built up area [ha]", "other [ha]"],
        )
        df["vegetation [ha]"] = veg
        df["built up area [ha]"] = bua
        df["other [ha]"] = other
        df["year"] = year

        df = df.set_index("year")

        st.bar_chart(df, width=700)