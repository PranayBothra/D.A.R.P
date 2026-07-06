import streamlit as st

st.set_page_config(
    page_title="ML Playground | From Scratch",
    layout="wide"
)

pages = {
    "": [
        st.Page("pages/Home.py", title="Home", icon=":material/home:"),
    ],
    "Modules": [
        st.Page("pages/Classification.py", title="Classification", icon=":material/category:"),
        st.Page("pages/Regression.py", title="Regression", icon=":material/show_chart:"),
        st.Page("pages/Clustering.py", title="Clustering", icon=":material/scatter_plot:"),
    ],
}

pg = st.navigation(pages)
pg.run()