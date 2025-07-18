import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from utils import *
from download_utils import get_download_link

st.set_page_config(page_title="Data Analysis App", layout="wide")

# Sidebar - File uploader
st.sidebar.header("📁 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df = replace_missing_values(df)  # Initial cleaning

        st.title("📊 Data Wrangling, EDA & ML Web App")

        # Tabs for organization
        tab1, tab2, tab3 = st.tabs(["🧹 Data Wrangling", "📈 Exploratory Data Analysis", "🤖 ML Model"])

        with tab1:
            st.subheader("🧹 Data Wrangling")

            # Rename Columns
            with st.expander("🔤 Rename Columns"):
                col_to_rename = st.selectbox("Select a column to rename", df.columns, key="rename_column")
                new_col_name = st.text_input("Enter new column name", value=col_to_rename, key="new_column_name")
                if st.button("Rename Column"):
                    df = df.rename(columns={col_to_rename: new_col_name})
                    st.success(f"Renamed '{col_to_rename}' to '{new_col_name}'")

            # Handle Missing Values
            with st.expander("❓ Handle Missing Values"):
                if st.button("Replace '?' with NaN"):
                    df.replace("?", np.nan, inplace=True)
                    st.success("Replaced '?' with NaN")

                if st.button("Impute Missing Numerical (Mean)"):
                    df = impute_numerical(df)
                    st.success("Numerical missing values replaced with mean")

                if st.button("Impute Missing Categorical (Mode)"):
                    df = impute_categorical(df)
                    st.success("Categorical missing values replaced with mode")

            # Drop Column & Reset Index
            with st.expander("🗑 Drop Column and Reset Index"):
                drop_col = st.selectbox("Select a column to drop", df.columns)
                if st.button("Drop Column"):
                    df.drop(columns=drop_col, inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    st.success(f"Dropped '{drop_col}' and reset index")

            # Convert Data Types
            with st.expander("🔁 Convert Data Types"):
                col_to_convert = st.selectbox("Select a column to convert", df.columns, key="convert_col")
                dtype = st.radio("Choose data type", ["int", "float", "str"])
                if st.button("Convert Data Type"):
                    df[col_to_convert] = df[col_to_convert].astype(dtype)
                    st.success(f"Converted '{col_to_convert}' to {dtype}")

            # Normalization & Standardization
            with st.expander("⚙️ Normalize / Standardize Data"):
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                col_norm = st.selectbox("Select column to normalize/standardize", numeric_cols)
                method = st.radio("Select method", ["Simple Scaling", "Min-Max", "Z-score"])
                if st.button("Apply Normalization/Standardization"):
                    if method == "Simple Scaling":
                        df[col_norm] = df[col_norm] / df[col_norm].max()
                    elif method == "Min-Max":
                        df[col_norm] = (df[col_norm] - df[col_norm].min()) / (df[col_norm].max() - df[col_norm].min())
                    elif method == "Z-score":
                        df[col_norm] = (df[col_norm] - df[col_norm].mean()) / df[col_norm].std()
                    st.success(f"Applied {method} to '{col_norm}'")

            # Binning
            with st.expander("📦 Binning"):
                bin_col = st.selectbox("Select column to bin", numeric_cols)
                bins = st.slider("Number of bins", min_value=2, max_value=10, value=4)
                if st.button("Apply Binning"):
                    df[bin_col + "_binned"] = pd.cut(df[bin_col], bins)
                    st.success(f"Binned '{bin_col}' into {bins} categories")

            # Show preview and download
            st.markdown("### 🔍 Preview Cleaned Data")
            st.dataframe(df.head())
            st.sidebar.markdown(get_download_link(df, filename="cleaned_data.csv"), unsafe_allow_html=True)

        with tab2:
            st.subheader("📈 Exploratory Data Analysis")
            st.write("Coming soon...")

        with tab3:
            st.subheader("🤖 ML Model Development")
            st.write("Coming soon...")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.sidebar.info("👈 Upload a CSV file to begin")