import os, io, json, math, random, time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, select, text, Index
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv

from flask import session

# Optional Redis (cache); app still works without Redis
try:
    from redis import Redis
except Exception:
    Redis = None  # type: ignore

load_dotenv()

# ---- Flask ----
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))

# ---- DB (SQLite) ----
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///quakes_month.db")
engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, future=True)

class Base(DeclarativeBase): pass

class Quake(Base):
    """
    Schema tailored to USGS 'all_month.csv'.
    Only the columns we need for this assignment are strongly typed/indexed.
    """
    __tablename__ = "all_month"

    id        = Column(String, primary_key=True)
    time      = Column(String, nullable=False)       # ISO8601 string
    latitude  = Column(Float,  nullable=True)
    longitude = Column(Float,  nullable=True)
    depth     = Column(Float,  nullable=True)
    mag       = Column(Float,  nullable=True)
    magType   = Column(String, nullable=True)
    nst       = Column(Integer, nullable=True)
    net       = Column(String, nullable=True)
    updated   = Column(String, nullable=True)
    place     = Column(String, nullable=True)
    type      = Column(String, nullable=True)

# Create table if missing (works with `flask --app app run`)
Base.metadata.create_all(engine)

# Predeclare indexes we want to time
IDX_SPECS = [
    ("ix_all_month_time",      "CREATE INDEX IF NOT EXISTS ix_all_month_time      ON all_month (time)"),
    ("ix_all_month_mag",       "CREATE INDEX IF NOT EXISTS ix_all_month_mag       ON all_month (mag)"),
    ("ix_all_month_net",       "CREATE INDEX IF NOT EXISTS ix_all_month_net       ON all_month (net)"),
    ("ix_all_month_place",     "CREATE INDEX IF NOT EXISTS ix_all_month_place     ON all_month (place)"),
    ("ix_all_month_lat_long",  "CREATE INDEX IF NOT EXISTS ix_all_month_lat_long  ON all_month (latitude, longitude)")
]

# ---- Redis (optional) ----
REDIS_URL = os.getenv("REDIS_URL", "").strip()
redis_client: Optional[Redis] = None
if REDIS_URL and Redis is not None:
    try:
        redis_client = Redis.from_url(REDIS_URL)
        # quick pong check
        redis_client.ping()
    except Exception:
        redis_client = None


# ---------- Helpers ----------

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance between (lat1,lon1) and (lat2,lon2) in km."""
    if None in (lat1, lon1, lat2, lon2):
        return float("inf")
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi   = math.radians(lat2 - lat1)
    dl     = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def canonical_key(prefix: str, payload: Dict) -> str:
    return f"{prefix}:{json.dumps(payload, sort_keys=True)}"

def cache_get(key: str) -> Optional[List[Dict]]:
    if not redis_client:
        return None
    try:
        raw = redis_client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None

def cache_set(key: str, value: List[Dict], ttl_sec: int = 300) -> None:
    if not redis_client:
        return
    try:
        redis_client.setex(key, ttl_sec, json.dumps(value))
    except Exception:
        pass


# ---------- CSV → DB load + timing ----------
def load_csv_to_db_with_timing(csv_path: str) -> Dict:
    """
    Create table (if missing), load CSV (replace contents), create indexes.
    Tolerant to missing columns: fills defaults when absent.
    Returns timing metrics: load_sec, index_secs{by name}, total_sec.
    """
    t0 = time.perf_counter()
    Base.metadata.create_all(engine)

    df = pd.read_csv(csv_path)

    # Case-insensitive column map
    cols = {c.lower(): c for c in df.columns}

    def has(name: str) -> bool:
        return name.lower() in cols

    def col(name: str):
        """Return a Series for the CSV column (case-insensitive)."""
        return df[cols[name.lower()]]

    def get_num(name: str):
        """Numeric series or all-NaN if missing."""
        return pd.to_numeric(col(name), errors="coerce") if has(name) else pd.Series([float("nan")] * len(df))

    def get_text(name: str, default_blank=True):
        """Text series or blanks if missing."""
        if has(name):
            return col(name).astype(str)
        return pd.Series(["" if default_blank else None] * len(df), dtype="object")

    # --- Build a normalized frame with graceful fallbacks ---
    # Required minimal set for your app: id, time, latitude, longitude, mag
    # Everything else is optional and will be NULL/blank if absent.
    time_series = (
        pd.to_datetime(get_text("time"), errors="coerce")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .fillna("")
    )

    updated_series = (
        pd.to_datetime(get_text("updated"), errors="coerce")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .fillna("")
    )

    df_norm = pd.DataFrame({
        "id":        get_text("id"),
        "time":      time_series,
        "latitude":  get_num("latitude"),
        "longitude": get_num("longitude"),
        "depth":     get_num("depth"),
        "mag":       get_num("mag"),
        "magType":   get_text("magType"),   # fine if missing → blanks
        "nst":       (pd.to_numeric(get_text("nst"), errors="coerce").astype("Int64")
                      if has("nst") else pd.Series([pd.NA] * len(df), dtype="Int64")),
        "net":       get_text("net"),
        "updated":   updated_series,
        "place":     get_text("place"),
        "type":      get_text("type"),
    })

    # --- Convert into ORM rows (None for NaN/blank) ---
    rows = []
    for _, r in df_norm.iterrows():
        pk = (r["id"] or "").strip()
        if not pk:
            continue  # skip rows with no primary key

        def none_if_nan(x):
            if isinstance(x, float) and math.isnan(x):
                return None
            if isinstance(x, pd._libs.missing.NAType):
                return None
            # treat textual "nan"/"NaN" as NULL for some fields
            if isinstance(x, str) and x.strip().lower() == "nan":
                return None
            return x

        rows.append(Quake(
            id=pk,
            time=r["time"] or "",
            latitude=none_if_nan(r["latitude"]),
            longitude=none_if_nan(r["longitude"]),
            depth=none_if_nan(r["depth"]),
            mag=none_if_nan(r["mag"]),
            magType=none_if_nan(r["magType"]),
            nst=(None if (pd.isna(r["nst"])) else int(r["nst"])),
            net=none_if_nan(r["net"]),
            updated=r["updated"] or "",
            place=none_if_nan(r["place"]),
            type=none_if_nan(r["type"]),
        ))

    # --- Replace table contents ---
    with SessionLocal() as s, s.begin():
        s.query(Quake).delete()
        if rows:
            s.add_all(rows)

    t_load = time.perf_counter()

    # --- Build indexes and time each (still fine if many NULLs) ---
    index_times = {}
    with engine.begin() as conn:
        for name, stmt in IDX_SPECS:
            t1 = time.perf_counter()
            conn.exec_driver_sql(stmt)
            index_times[name] = time.perf_counter() - t1

    total = time.perf_counter() - t0
    return {
        "row_count": len(rows),
        "load_seconds": t_load - t0,
        "index_seconds": index_times,
        "total_seconds": total
    }

# ---------- Routes ----------

@app.route("/")
def index():
    # Counts may fail if table empty; handle nicely.
    total = 0
    try:
        with SessionLocal() as s:
            total = s.scalar(select(text("COUNT(*)")).select_from(Quake)) or 0
    except OperationalError:
        pass
    caching = bool(redis_client)
    return render_template("index.html", total=total, caching=caching)


@app.route("/initdb", methods=["GET","POST"])
def initdb():
    """
    Upload all_month.csv; create table; time load and index builds. Show metrics.
    """
    metrics = None
    if request.method == "POST":
        f = request.files.get("csv_file")
        if not f or f.filename == "":
            flash("Choose the USGS CSV file (all_month.csv).", "error")
            return redirect(request.url)
        try:
            data = f.read()
            tmp = "_upload_all_month.csv"
            with open(tmp, "wb") as out:
                out.write(data)
            metrics = load_csv_to_db_with_timing(tmp)
            os.remove(tmp)
            flash(f"Loaded {metrics['row_count']} rows.", "success")
        except Exception as e:
            flash(f"Init failed: {e}", "error")
            return redirect(request.url)
    return render_template("initdb.html", metrics=metrics)


# ---- 1) Batch of N random queries (by random IDs) ----
@app.route("/bench-random", methods=["GET","POST"])
def bench_random():
    """
    Up to 1000 random single-row lookups (by id), with optional Redis caching
    and the ability to reuse the same ID set to demonstrate cache benefits.
    """
    results = []
    stats = None
    form = {"n": "100", "use_cache": "on", "reuse_ids": ""}

    if request.method == "POST":
        # read form
        n_str = (request.form.get("n") or "100").strip()
        n = max(1, min(1000, int(n_str)))
        use_cache = (request.form.get("use_cache") == "on")
        reuse_ids = (request.form.get("reuse_ids") == "on")

        form["n"] = str(n)
        form["use_cache"] = "on" if use_cache else ""
        form["reuse_ids"] = "on" if reuse_ids else ""

        # 1) pick IDs
        with SessionLocal() as s:
            if reuse_ids and session.get("random_ids"):
                ids = session["random_ids"][:n]
            else:
                # fetch N random ids in one SQL call
                ids = [row[0] for row in s.execute(
                    text("SELECT id FROM all_month ORDER BY RANDOM() LIMIT :n"),
                    {"n": n}
                ).fetchall()]
                # stash in session so next run can reuse the exact same set
                session["random_ids"] = ids

        # 2) run lookups (cached or not)
        t0 = time.perf_counter()
        rows = []
        cache_hits = 0
        cache_misses = 0

        for qid in ids:
            row_dict = None

            if use_cache and redis_client:
                # try cache first (per-id cache)
                ck = f"row:{qid}"
                cached = cache_get(ck)
                if cached:
                    # cache_get returns a list when we stored list; store single row as list of one
                    # Keep backward compatible: accept dict too.
                    if isinstance(cached, list) and cached and isinstance(cached[0], dict):
                        row_dict = cached[0]
                    elif isinstance(cached, dict):
                        row_dict = cached
                if row_dict:
                    cache_hits += 1

            if row_dict is None:
                # miss → fetch from DB
                with SessionLocal() as s:
                    rec = s.execute(
                        text("SELECT time, latitude, longitude, id, mag FROM all_month WHERE id=:id"),
                        {"id": qid}
                    ).fetchone()
                if rec:
                    row_dict = {"time": rec[0], "latitude": rec[1], "longitude": rec[2], "id": rec[3], "mag": rec[4]}
                    cache_misses += 1
                    # write-through cache
                    if use_cache and redis_client:
                        cache_set(f"row:{qid}", [row_dict], ttl_sec=600)

            if row_dict:
                rows.append(row_dict)

        elapsed = time.perf_counter() - t0

        stats = {
            "count": len(rows),
            "total_sec": elapsed,
            "avg_ms": (elapsed / len(rows) * 1000) if rows else 0,
            "cached": use_cache and bool(redis_client),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        }
        results = rows[:25]  # show a small sample

    return render_template("bench_random.html", form=form, stats=stats, results=results)

# ---- 2) Restricted queries (place, distance, time range, mag range) + timing + Redis cache ----
@app.route("/bench-restricted", methods=["GET","POST"])
def bench_restricted():
    """
    Run batch queries under constraints; measure time, optionally use Redis cache.
    """
    results = []
    stats = None
    form = {"mode":"place_ca","n":"100","use_cache":"on","lat":"","lon":"","radius_km":"","t_start":"","t_end":"","m_low":"","m_high":""}

    if request.method == "POST":
        mode = request.form.get("mode") or "place_ca"
        n = max(1, min(1000, int(request.form.get("n") or "100")))
        use_cache = (request.form.get("use_cache") == "on")

        form.update({
            "mode": mode, "n": str(n), "use_cache": "on" if use_cache else "",
            "lat": request.form.get("lat") or "",
            "lon": request.form.get("lon") or "",
            "radius_km": request.form.get("radius_km") or "",
            "t_start": request.form.get("t_start") or "",
            "t_end": request.form.get("t_end") or "",
            "m_low": request.form.get("m_low") or "",
            "m_high": request.form.get("m_high") or ""
        })

        # Build a canonical parameter set for cache key
        params = dict(
            mode=mode, n=n,
            lat=form["lat"], lon=form["lon"], radius_km=form["radius_km"],
            t_start=form["t_start"], t_end=form["t_end"],
            m_low=form["m_low"], m_high=form["m_high"]
        )
        cache_key = canonical_key("restricted", params)

        cached = cache_get(cache_key) if use_cache else None
        if cached is not None:
            results = cached
            stats = {"count": len(results), "total_sec": 0.0, "avg_ms": 0.0, "cached": True}
            return render_template("bench_restricted.html", form=form, stats=stats, results=results[:50])

        # Otherwise compute
        with SessionLocal() as s:
            t0 = time.perf_counter()
            rows: List[Dict] = []

            if mode == "place_ca":
                # rows where place contains " CA"
                base = s.execute(
                    text("SELECT time, latitude, longitude, id, mag, place FROM all_month WHERE place LIKE :p ORDER BY RANDOM() LIMIT :n"),
                    {"p": "% CA%", "n": n}
                ).fetchall()
                rows = [{"time": r[0], "latitude": r[1], "longitude": r[2], "id": r[3], "mag": r[4], "place": r[5]} for r in base]

            elif mode == "distance":
                # bounding box first, then precise Haversine in Python
                lat0 = float(form["lat"]); lon0 = float(form["lon"]); radius = float(form["radius_km"])
                delta_lat = radius / 111.0
                delta_lon = radius / (111.0 * max(math.cos(math.radians(lat0)), 1e-6))
                bb = s.execute(
                    text("""SELECT time, latitude, longitude, id, mag, place
                            FROM all_month
                            WHERE latitude BETWEEN :lat0-:dlat AND :lat0+:dlat
                              AND longitude BETWEEN :lon0-:dlon AND :lon0+:dlon
                          """),
                    {"lat0": lat0, "lon0": lon0, "dlat": delta_lat, "dlon": delta_lon}
                ).fetchall()
                # compute precise distance and take up to n
                filtered = []
                for r in bb:
                    d = _haversine_km(lat0, lon0, r[1], r[2])
                    if d <= radius:
                        filtered.append({"time": r[0], "latitude": r[1], "longitude": r[2], "id": r[3], "mag": r[4], "place": r[5], "km": round(d,2)})
                        if len(filtered) >= n:
                            break
                rows = filtered

            elif mode == "time_range":
                # assumes form: YYYY-MM-DD HH:MM:SS strings (our loader normalized to this)
                base = s.execute(
                    text("""SELECT time, latitude, longitude, id, mag, place
                            FROM all_month
                            WHERE time BETWEEN :t0 AND :t1
                            ORDER BY RANDOM() LIMIT :n"""),
                    {"t0": form["t_start"], "t1": form["t_end"], "n": n}
                ).fetchall()
                rows = [{"time": r[0], "latitude": r[1], "longitude": r[2], "id": r[3], "mag": r[4], "place": r[5]} for r in base]

            elif mode == "mag_range":
                m_low  = float(form["m_low"]) if form["m_low"] else None
                m_high = float(form["m_high"]) if form["m_high"] else None
                where = []
                params2 = {}
                if m_low is not None:  where.append("mag >= :mlow");  params2["mlow"] = m_low
                if m_high is not None: where.append("mag <= :mhigh"); params2["mhigh"] = m_high
                sql = "SELECT time, latitude, longitude, id, mag, place FROM all_month"
                if where: sql += " WHERE " + " AND ".join(where)
                sql += " ORDER BY RANDOM() LIMIT :n"
                params2["n"] = n
                base = s.execute(text(sql), params2).fetchall()
                rows = [{"time": r[0], "latitude": r[1], "longitude": r[2], "id": r[3], "mag": r[4], "place": r[5]} for r in base]

            elapsed = time.perf_counter() - t0

        stats = {"count": len(rows), "total_sec": elapsed, "avg_ms": (elapsed/len(rows)*1000) if rows else 0, "cached": False}
        results = rows[:50]

        if use_cache:
            cache_set(cache_key, rows, ttl_sec=300)

    return render_template("bench_restricted.html", form=form, stats=stats, results=results)


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    app.run(debug=bool(int(os.getenv("FLASK_DEBUG","1"))))
