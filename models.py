from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import pandas as pd
import os
import pytz
from datetime import datetime
from .report_logic import calculate_uptime_downtime  # your existing logic

DATABASE_URL = "your_postgres_url_here"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class StoreStatus(Base):
    __tablename__ = "store_status"
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(String, index=True)
    timestamp_utc = Column(DateTime)
    status = Column(String)

class MenuHours(Base):
    __tablename__ = "business_hours"
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(String, index=True)
    day_of_week = Column(Integer)
    start_time_local = Column(String)
    end_time_local = Column(String)

class Timezone(Base):
    __tablename__ = "timezone"
    id = Column(Integer, primary_key=True, index=True)
    store_id = Column(String, index=True)
    timezone_str = Column(String)

class Report(Base):
    __tablename__ = "reports"
    report_id = Column(String, primary_key=True, index=True)
    status = Column(String, default="Running")
    created_at = Column(DateTime, default=datetime.utcnow)
    file_path = Column(String, nullable=True)

Base.metadata.create_all(bind=engine)

def ingest_data():
    db: Session = SessionLocal()
    try:
        status_df = pd.read_csv("data/store_status.csv")
        for _, row in status_df.iterrows():
            db.merge(StoreStatus(
                store_id=row["store_id"],
                timestamp_utc=pd.to_datetime(row["timestamp_utc"]),
                status=row["status"]
            ))

        if os.path.exists("data/menu_hours.csv"):
            hours_df = pd.read_csv("data/menu_hours.csv")
            for _, row in hours_df.iterrows():
                db.merge(MenuHours(
                    store_id=row["store_id"],
                    day_of_week=int(row["day_of_week"]),
                    start_time_local=row["start_time_local"],
                    end_time_local=row["end_time_local"]
                ))

        if os.path.exists("data/timezones.csv"):
            tz_df = pd.read_csv("data/timezones.csv")
            for _, row in tz_df.iterrows():
                db.merge(Timezone(
                    store_id=row["store_id"],
                    timezone_str=row["timezone_str"]
                ))
        db.commit()
    finally:
        db.close()

def generate_report(report_id: str):
    db: Session = SessionLocal()
    try:
        max_timestamp = db.query(StoreStatus.timestamp_utc).order_by(StoreStatus.timestamp_utc.desc()).first()
        current_time = max_timestamp[0] if max_timestamp else datetime.utcnow().replace(tzinfo=pytz.UTC)

        store_ids = [s[0] for s in db.query(StoreStatus.store_id).distinct().all()]
        report_data = []

        for store_id in store_ids:
            status_data = pd.read_sql(
                db.query(StoreStatus).filter(StoreStatus.store_id == store_id).statement,
                db.connection()
            )
            status_data["timestamp_utc"] = pd.to_datetime(status_data["timestamp_utc"])
            business_hours = get_business_hours(db, store_id)
            timezone_str = get_timezone(db, store_id)

            metrics = calculate_uptime_downtime(
                store_id, status_data, business_hours, timezone_str, current_time
            )
            report_data.append(metrics)

        os.makedirs("reports", exist_ok=True)
        file_path = f"reports/{report_id}.csv"
        pd.DataFrame(report_data).to_csv(file_path, index=False)

        report = db.query(Report).filter(Report.report_id == report_id).first()
        report.status = "Complete"
        report.file_path = file_path
        db.commit()
    except Exception:
        report = db.query(Report).filter(Report.report_id == report_id).first()
        report.status = "Failed"
        db.commit()
    finally:
        db.close()

def get_report_status(report_id: str, db: Session):
    report = db.query(Report).filter(Report.report_id == report_id).first()
    if not report:
        return "Not Found", None
    return report.status, report.file_path
