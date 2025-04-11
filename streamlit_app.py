import streamlit as st
import pandas as pd
from datetime import datetime
from io import StringIO

# Set page configuration
st.set_page_config(page_title="Data Dashboard", layout="wide")

# App title and subtitle
st.title("📊 Data Dashboard")
st.markdown("""
Welcome to the Data Dashboard. Upload a CSV or Excel file to view, analyze and interact with your data.
""")

# File uploader
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Load file into DataFrame
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ File uploaded and parsed successfully!")

        # Show top-level metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📁 Rows", f"{df.shape[0]:,}")
        col2.metric("📄 Columns", f"{df.shape[1]:,}")
        col3.metric("🕒 Current Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        col4.metric("🔍 Null Values", f"{df.isnull().sum().sum():,}")

        st.markdown("---")

        # Interactive data preview
        with st.expander("🔍 View DataFrame"):
            st.dataframe(df, use_container_width=True)

        # Column-wise analysis
        st.subheader("📈 Column-wise Summary")
        for column in df.select_dtypes(include=["object", "category"]).columns:
            st.markdown(f"### 🗂️ {column}")
            st.write(df[column].value_counts())

        for column in df.select_dtypes(include=["int", "float"]).columns:
            st.markdown(f"### 📊 {column}")
            st.bar_chart(df[column].value_counts().head(10))

        st.markdown("---")

        # Filter data by column
        st.subheader("🔎 Filter Data")
        selected_column = st.selectbox("Select column to filter", df.columns)
        unique_values = df[selected_column].dropna().unique()

        selected_value = st.selectbox("Select value", unique_values)
        filtered_df = df[df[selected_column] == selected_value]

        st.markdown(f"### Filtered Data by `{selected_column} == {selected_value}`")
        st.dataframe(filtered_df, use_container_width=True)

        st.markdown("---")

        # Download filtered data
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Filtered Data as CSV", csv, "filtered_data.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("📥 Upload a CSV or Excel file to begin.")
