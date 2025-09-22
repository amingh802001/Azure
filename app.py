import os
import io
from typing import Optional
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from sqlalchemy import create_engine, Column, String, Float, Integer, select, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from dotenv import load_dotenv

load_dotenv()

# --- Flask setup ---
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))

# --- Database setup (SQLite) ---
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///quakes.db")
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, future=True)

class Base(DeclarativeBase):
    pass

class Earthquake(Base):
    __tablename__ = "earthquakes"
    # Expected CSV columns: time, lat, long, mag, nst, net, id
    id   = Column(String, primary_key=True)   # use provided id as PK
    time = Column(String, nullable=False)     # store as text "YYYY-MM-DD HH:MM:SS"
    lat  = Column(Float,  nullable=True)
    long = Column(Float,  nullable=True)
    mag  = Column(Float,  nullable=True)
    nst  = Column(Integer, nullable=True)
    net  = Column(String, nullable=True)

# ✅ Ensure tables exist at import time (works with `flask --app app run`)
Base.metadata.create_all(engine)

def init_db_from_csv(csv_path: str) -> int:
    """Create tables (idempotent) and load CSV rows. Returns number of rows loaded."""
    Base.metadata.create_all(engine)
    df = pd.read_csv(csv_path)

    expected = {"time","lat","long","mag","nst","net","id"}
    # case-insensitive check
    cols_lower = {c.lower(): c for c in df.columns}
    missing = [c for c in expected if c not in cols_lower]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    def col(name): return df[cols_lower[name]]

    df_norm = pd.DataFrame({
        "id":   col("id").astype(str),
        "time": pd.to_datetime(col("time"), errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S").fillna(""),
        "lat":  pd.to_numeric(col("lat"), errors="coerce"),
        "long": pd.to_numeric(col("long"), errors="coerce"),
        "mag":  pd.to_numeric(col("mag"), errors="coerce"),
        "nst":  pd.to_numeric(col("nst"), errors="coerce").astype("Int64"),
        "net":  col("net").astype(str),
    })

    rows = []
    for _, r in df_norm.iterrows():
        if not r["id"]:
            continue  # skip if no primary key
        rows.append(Earthquake(
            id=r["id"],
            time=r["time"] or "",
            lat=None if pd.isna(r["lat"]) else float(r["lat"]),
            long=None if pd.isna(r["long"]) else float(r["long"]),
            mag=None if pd.isna(r["mag"]) else float(r["mag"]),
            nst=None if pd.isna(r["nst"]) else int(r["nst"]),
            net=(None if r["net"] in ("nan", "NaN", "") else r["net"])
        ))

    with SessionLocal() as s, s.begin():
        # Simple reset before load (clear table)
        s.query(Earthquake).delete()
        s.add_all(rows)
    return len(rows)

# ---------- Routes ----------
@app.route("/")
def index():
    with SessionLocal() as s:
        total = s.scalar(select(text("COUNT(*)")).select_from(Earthquake))
    return render_template("index.html", total=total)

@app.route("/initdb", methods=["GET","POST"])
def initdb():
    """Upload CSV via form and load into SQLite."""
    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or f.filename == "":
            flash("Choose a CSV file.", "error")
            return redirect(request.url)
        try:
            data = f.read()
            tmp_path = "_upload.csv"
            with open(tmp_path, "wb") as out:
                out.write(data)
            n = init_db_from_csv(tmp_path)
            os.remove(tmp_path)
            flash(f"Database initialized with {n} rows.", "success")
            return redirect(url_for("index"))
        except Exception as e:
            flash(f"Init failed: {e}", "error")
            return redirect(request.url)
    return render_template("initdb.html")

# (1) Magnitude range: show time, latitude, longitude, id
@app.route("/mag", methods=["GET", "POST"])
def mag_range():
    results = []
    form = {"low":"", "high":""}
    if request.method == "POST":
        form["low"]  = (request.form.get("low")  or "").strip()
        form["high"] = (request.form.get("high") or "").strip()
        try:
            low  = float(form["low"])  if form["low"]  else None
            high = float(form["high"]) if form["high"] else None
        except ValueError:
            flash("Magnitude must be numeric.", "error")
            return render_template("mag.html", form=form, results=results)

        with SessionLocal() as s:
            q = s.query(Earthquake)
            if low  is not None: q = q.filter(Earthquake.mag >= low)
            if high is not None: q = q.filter(Earthquake.mag <= high)
            rows = q.order_by(Earthquake.mag).all()
            results = [{"time": r.time, "lat": r.lat, "long": r.long, "id": r.id, "mag": r.mag} for r in rows]
    return render_template("mag.html", form=form, results=results)

# (2) Specify a net value: count occurrences, delete, show remaining
@app.route("/net-delete", methods=["GET","POST"])
def net_delete():
    info = None
    if request.method == "POST":
        net = (request.form.get("net") or "").strip()
        if not net:
            flash("Enter a net value.", "error")
            return render_template("net_delete.html", info=info)
        with SessionLocal() as s, s.begin():
            to_delete = s.query(Earthquake).filter(Earthquake.net == net).count()
            s.query(Earthquake).filter(Earthquake.net == net).delete()
            remaining = s.query(Earthquake).count()
        info = {"net": net, "deleted": to_delete, "remaining": remaining}
        flash(f"Deleted {to_delete} row(s) with net='{net}'. Remaining: {remaining}.", "success")
    return render_template("net_delete.html", info=info)

# (3) Enter a net id (interpreted as quake id) OR a time; modify attributes
@app.route("/edit", methods=["GET","POST"])
def edit_quake():
    state = {"step": "find", "id":"", "time":"", "record": None}
    if request.method == "POST":
        step = request.form.get("step","find")
        if step == "find":
            quake_id = (request.form.get("id") or "").strip()
            t        = (request.form.get("time") or "").strip()
            with SessionLocal() as s:
                q = s.query(Earthquake)
                rec = q.filter(Earthquake.id == quake_id).first() if quake_id else None
                if not rec and t:
                    rec = q.filter(Earthquake.time == t).first()
                if not rec:
                    flash("No matching record found.", "error")
                    return render_template("edit.html", state=state)
                state = {"step":"edit","id":rec.id,"time":rec.time,"record":rec}
                return render_template("edit.html", state=state)

        elif step == "edit":
            quake_id = request.form.get("orig_id")
            with SessionLocal() as s, s.begin():
                rec = s.query(Earthquake).filter(Earthquake.id == quake_id).first()
                if not rec:
                    flash("Record disappeared; try again.", "error")
                    return redirect(url_for("edit_quake"))
                # Update any provided attribute (blank = leave as-is)
                def val(name): return (request.form.get(name) or "").strip()
                new_time = val("time");  new_lat = val("lat");  new_long = val("long")
                new_mag  = val("mag");   new_nst = val("nst");  new_net  = val("net")
                if new_time: rec.time = new_time
                if new_lat:  rec.lat  = float(new_lat)
                if new_long: rec.long = float(new_long)
                if new_mag:  rec.mag  = float(new_mag)
                if new_nst:  rec.nst  = int(new_nst)
                if new_net:  rec.net  = new_net
            flash("Record updated.", "success")
            return redirect(url_for("edit_quake"))
    return render_template("edit.html", state=state)

if __name__ == "__main__":
    # Not strictly needed anymore, but harmless
    Base.metadata.create_all(engine)
    app.run(debug=bool(int(os.getenv("FLASK_DEBUG","1"))))
