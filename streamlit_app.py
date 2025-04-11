import streamlit as st
import pandas as pd
import os
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from models import SessionLocal, Report, get_report_status, generate_report, ingest_data

st.set_page_config(page_title="Loop XYZ Store Dashboard", layout="wide")

st.title("📊 Loop XYZ - Store Monitoring Dashboard")

# Sidebar for Upload and Trigger
with st.sidebar:
    st.header("🔄 Data Ingestion")
    if st.button("Ingest Latest Data"):
        try:
            ingest_data()
            st.success("✅ Data ingested successfully.")
        except Exception as e:
            st.error(f"❌ Ingestion failed: {str(e)}")

    st.header("📁 Trigger Report")
    if st.button("Generate Report"):
        report_id = str(uuid.uuid4())
        db: Session = SessionLocal()
        try:
            db.add(Report(report_id=report_id, status="Running", created_at=datetime.utcnow()))
            db.commit()
            generate_report(report_id)
            st.session_state['latest_report_id'] = report_id
            st.success(f"✅ Report triggered. Report ID: {report_id}")
        finally:
            db.close()

# Main Section: Report Viewer
st.header("📈 Report Viewer")

report_id = st.text_input("Enter Report ID to Check:", value=st.session_state.get('latest_report_id', ''))

if report_id:
    db: Session = SessionLocal()
    try:
        status, file_path = get_report_status(report_id, db)
        if status == "Running":
            st.info("⏳ Report is still being generated. Please check back shortly.")
        elif status == "Complete" and os.path.exists(file_path):
            st.success("✅ Report is ready!")
            df = pd.read_csv(file_path)
            st.download_button("⬇️ Download Report", data=open(file_path, "rb"), file_name=f"report_{report_id}.csv")
            st.dataframe(df, use_container_width=True)
        else:
            st.error("❌ Report generation failed or file not found.")
    finally:
        db.close()
