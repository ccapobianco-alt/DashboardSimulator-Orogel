import threading, webbrowser
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.dash_table import DataTable

# ───────────────────────── Config
COMPANY_NAME = "OROGEL SCA IN SIGLA A.C.O. SCA"
AREA_TOTALE_HA = 800

CROP_ALL = "Tutte le colture"
CROPS = [CROP_ALL, "Pisello", "Spinacio", "Fagiolino"]  # Solo 3 colture

SENS = {  # sensibilità per fattore resa annuale
    "Pisello":   {"heat": 0.08, "frost": 0.06, "wet": 0.04},
    "Spinacio":  {"heat": 0.13, "frost": 0.03, "wet": 0.05},
    "Fagiolino": {"heat": 0.11, "frost": 0.02, "wet": 0.06},
}

MONTHS_IT = ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"]
BAR_COLOR = "#6d28d9"  # viola per barre

YEARS = list(range(2020, 2026))
DEFAULT_YEAR = YEARS[-1]
DEFAULT_CROP = CROP_ALL

# ─ Utils
def tight_range(values, pad: float = 0.05):
    s = pd.Series(values)
    if s.empty or not np.isfinite(s.min()) or not np.isfinite(s.max()):
        return None
    vmin, vmax = float(s.min()), float(s.max())
    if vmin == vmax: return (vmin - 1, vmax + 1)
    span = vmax - vmin
    return (vmin - pad*span, vmax + pad*span)

def y_tickformat(values) -> str:
    s = pd.Series(values).astype(float)
    if s.empty or not np.isfinite(s.max()): return ",.0f"
    return "~s" if float(s.max()) >= 1000 else ",.0f"

def fmt_int(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")

def _distribute_month_total(total_q: int, n_days: int, rng: np.random.Generator) -> np.ndarray:
    if n_days == 0 or total_q <= 0: return np.zeros(n_days, dtype=int)
    weights = rng.dirichlet(np.full(n_days, 2.0))
    daily = np.floor(weights * total_q).astype(int)
    diff = total_q - daily.sum()
    if diff > 0: daily[:diff] += 1
    return daily

# ─ Stats robuste
def safe_corr(a, b) -> float:
    """Pearson senza warning: 0 se varianza nulla o dati insufficienti."""
    x = pd.Series(a, dtype="float64")
    y = pd.Series(b, dtype="float64")
    xy = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(xy) < 2: return 0.0
    sx = float(xy.iloc[:, 0].std(ddof=0)); sy = float(xy.iloc[:, 1].std(ddof=0))
    if sx == 0.0 or sy == 0.0: return 0.0
    cx = xy.iloc[:, 0] - xy.iloc[:, 0].mean()
    cy = xy.iloc[:, 1] - xy.iloc[:, 1].mean()
    return float((cx * cy).mean() / (sx * sy))

# ─ Meteo & pH
def monthly_climate_random(year: int, rng: np.random.Generator):
    m = np.arange(1, 13)
    # Temperature
    t_mean = rng.normal(14.5, 0.8)
    t_amp  = np.clip(rng.normal(10.5, 1.2), 8.5, 12.5)
    t_phi  = rng.uniform(-0.3, 0.3)
    temp_m = t_mean + t_amp*np.cos(2*np.pi*((m - (8 + t_phi))/12.0)) + rng.normal(0, 0.6, 12)
    temp_m = np.clip(temp_m, -2.0, 28.0)
    # Umidità
    h_base = rng.normal(70, 2.5)
    h_amp  = np.clip(rng.normal(9.0, 2.0), 6.0, 12.0)
    h_phi  = rng.uniform(-0.2, 0.2)
    hum_m  = h_base + h_amp*np.cos(2*np.pi*((m - (1 + h_phi))/12.0)) + rng.normal(0, 1.5, 12)
    hum_m  = np.clip(hum_m, 45, 95)
    # Piogge
    annual_mm = np.clip(rng.normal(750, 100), 550, 950)
    gauss = lambda x, mu, s: np.exp(-0.5*((x-mu)/s)**2)
    w = 0.9*gauss(m, 4.0, 1.2) + 1.1*gauss(m, 11.0, 1.0) + 0.35
    w *= rng.lognormal(0.0, 0.15, 12)
    w = np.clip(w, 0.05, None); w /= w.sum()
    rain_m = np.round(annual_mm * w, 1)
    # pH
    base_ph = np.clip(rng.normal(6.8, 0.25), 5.5, 8.0)
    rain_norm = rain_m / (np.mean(rain_m) if np.mean(rain_m) > 0 else 1.0)
    season = 0.12 * np.cos(2*np.pi*(m - 5)/12.0)
    rain_eff = -0.25 * (rain_norm - 1.0)
    ph_m = base_ph + season + rain_eff + rng.normal(0, 0.03, 12)
    ph_m = np.clip(ph_m, 5.2, 8.2)
    return temp_m.tolist(), hum_m.tolist(), rain_m.tolist(), ph_m.tolist()

def _daily_series_from_monthly(monthly_vals: list, dates: pd.DatetimeIndex) -> np.ndarray:
    year = dates[0].year
    idx_m = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="MS")
    vals  = monthly_vals + [monthly_vals[0]]
    s     = pd.Series(vals, index=idx_m)
    s_day = s.reindex(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")).interpolate("time")
    return s_day.rolling(15, center=True, min_periods=1).mean().values

def genera_meteo_anno(year: int) -> pd.DataFrame:
    rng = np.random.default_rng(year)  # seme per anno
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    t_m, h_m, r_m, ph_m = monthly_climate_random(year, rng)
    temp = np.clip(_daily_series_from_monthly(t_m, idx) + rng.normal(0, np.interp(idx.month,[1,6,12],[3.5,2.2,3.2]), len(idx)), -8, 42).round(1)
    hum  = np.clip(_daily_series_from_monthly(h_m, idx) + rng.normal(0, 4, len(idx)), 40, 98).round(1)
    rain = np.zeros(len(idx))
    for m in range(1, 13):
        mask = (idx.month == m); n = mask.sum()
        month_total = r_m[m-1]; mean_day = month_total / n
        norm = month_total / max(r_m); p_dry = np.clip(0.70 - 0.40*norm + rng.normal(0, 0.03), 0.15, 0.80)
        r = rng.exponential(mean_day, size=n); r[rng.random(n) < p_dry] = 0.0
        rain[mask] = np.round(r, 1)
    ph_daily = _daily_series_from_monthly(ph_m, idx) + rng.normal(0, 0.03, len(idx))
    ph_daily = np.clip(ph_daily, 5.0, 8.5).round(2)
    return pd.DataFrame({"Data": idx, "Anno": year, "Mese": idx.month.astype(int), "Giorno": idx.day.astype(int),
                         "Temperatura": temp, "Umidità": hum, "Precipitazioni": rain, "pH": ph_daily})

# ─ Variabilità di anno (macro-shock)
def year_macro_factor(year: int) -> float:
    """Fattore annata condiviso tra colture (media ~1)."""
    rng = np.random.default_rng(year * 7919 + 17)
    # più variabilità ma non estrema
    return float(np.clip(rng.normal(1.0, 0.16), 0.80, 1.24))

# ─ Specifiche colturali (random per anno, più variabili)
def genera_specs_colture(year: int, crops: list, area_tot: int, rng: np.random.Generator = None):
    rng = rng or np.random.default_rng(year + 13)
    resa_ranges = {"Pisello": (75, 95), "Spinacio": (85, 110), "Fagiolino": (85, 105)}
    # Dirichlet meno "piatta" → più variabilità nelle aree
    alpha  = np.ones(len(crops) - 1) * 2.8
    shares = rng.dirichlet(alpha)
    areas  = np.floor(shares * area_tot).astype(int)
    areas[-1] += area_tot - int(areas.sum())  # pareggio

    def random_weights():
        months = np.arange(1, 13, dtype=float)
        z = np.zeros(12, dtype=float)
        for _ in range(rng.integers(2, 4)):  # 2–3 picchi stagionali
            mu = rng.uniform(1, 12); s = rng.uniform(0.7, 1.7)
            z += np.exp(-0.5*((months - mu)/s)**2)
        z += 0.20
        z *= rng.lognormal(0.0, 0.18, 12)  # più irregolare
        z = np.clip(z, 1e-6, None)
        return (z / z.sum()).tolist()

    specs, i = {}, 0
    for crop in crops:
        if crop == CROP_ALL: continue
        lo, hi = resa_ranges[crop]
        # piccola deriva annua sulla resa media (±7% circa)
        mean = float(rng.uniform(lo, hi)) * float(np.clip(rng.normal(1.0, 0.07), 0.88, 1.14))
        sd   = float(mean * rng.uniform(0.10, 0.17))  # sd leggermente più ampia
        specs[crop] = {"area_ha": int(areas[i]), "resa_mean": round(mean,1), "resa_sd": round(sd,1), "w": random_weights()}
        i += 1

    delta = area_tot - sum(v["area_ha"] for v in specs.values())
    if delta:
        big = max(specs, key=lambda k: specs[k]["area_ha"])
        specs[big]["area_ha"] += delta
    return specs

# ─ Produzione
def _normalize_12(w):
    a = np.clip(np.array(w, dtype=float), 0, None)
    return (a/a.sum()) if a.sum() > 0 else np.ones(12)/12

def _climate_factor_for_crop(crop: str, temp: np.ndarray, rain: np.ndarray) -> float:
    days = max(len(temp), 1)
    hot   = (temp > 33).sum()/days
    frost = (temp < 0).sum()/days
    wet   = (rain > 12).sum()/days
    gdd10 = np.maximum(temp - 10.0, 0).sum()/(days*10.0)
    s = SENS[crop]
    base = 1.0 - (s["heat"]*hot + s["frost"]*frost + s["wet"]*wet) + 0.02*(gdd10 - 1.0)
    # banda un po' più ampia
    return float(np.clip(base, 0.80, 1.12))

def _seed_from_year_crop(dates_like, crop_name: str) -> int:
    year = int(pd.to_datetime(pd.Series(dates_like)).dt.year.iloc[0])
    return (year * 1009) ^ (abs(hash(crop_name)) & 0xFFFFFFFF)

def produzione_daily_per_coltura(dates, temp, rain, area_ha, resa_mean, resa_sd, weights_mensili, crop_name):
    rng = np.random.default_rng(_seed_from_year_crop(dates, crop_name))
    dates  = pd.to_datetime(dates)
    months = pd.DatetimeIndex(dates).month.astype(int).to_numpy()

    resa_base = float(np.clip(rng.normal(resa_mean, resa_sd), resa_mean - 2*resa_sd, resa_mean + 2*resa_sd))

    # fattori extra per aumentare variabilità (annata + coltura specifica)
    year  = int(pd.to_datetime(dates).dt.year.iloc[0])
    macro = year_macro_factor(year)                                  # annata
    crop_shock = float(np.clip(rng.normal(1.0, 0.12), 0.82, 1.22))   # shock specifico coltura

    resa_q_ha = (
        resa_base
        * _climate_factor_for_crop(crop_name, np.asarray(temp), np.asarray(rain))
        * float(rng.normal(0.98, 0.03))  # piccola alea residua
        * macro
        * crop_shock
    )
    annual_q  = int(round(area_ha * resa_q_ha))

    # pesi mensili ancora più irregolari
    w = _normalize_12(weights_mensili) * rng.lognormal(0.0, 0.16, 12)
    w = w / w.sum()
    month_totals = np.floor(annual_q * w).astype(int)

    prod_q = np.zeros(len(dates), dtype=int)
    for m in range(1, 13):
        mask = (months == m); n = int(mask.sum()); m_total = int(month_totals[m-1])
        daily = _distribute_month_total(m_total, n, rng)
        r = np.asarray(rain)[mask]; t = np.asarray(temp)[mask]
        penalty = np.ones(n); penalty[r > 12] *= 0.90; penalty[(t < 2) | (t > 34)] *= 0.92
        adj = np.floor(daily * penalty).astype(int)
        diff = m_total - adj.sum()
        if diff > 0 and len(adj) > 0: adj[:diff] += 1
        prod_q[mask] = adj
    return prod_q

def genera_df_anno_coltura(year: int, crop: str, meteo_df: pd.DataFrame, specs_year: dict) -> pd.DataFrame:
    dates = pd.to_datetime(meteo_df["Data"])
    temp  = meteo_df["Temperatura"].to_numpy()
    rain  = meteo_df["Precipitazioni"].to_numpy()
    if crop == CROP_ALL:
        total = np.zeros(len(meteo_df), dtype=int)
        for name, sp in specs_year.items():
            total += produzione_daily_per_coltura(dates, temp, rain, sp["area_ha"], sp["resa_mean"], sp["resa_sd"], sp["w"], name)
        prod, area = total, sum(v["area_ha"] for v in specs_year.values())
    else:
        sp   = specs_year[crop]
        prod = produzione_daily_per_coltura(dates, temp, rain, sp["area_ha"], sp["resa_mean"], sp["resa_sd"], sp["w"], crop)
        area = sp["area_ha"]
    df = meteo_df.copy()
    df["Produzione"] = prod.astype(int)
    df["Area_ha"]    = int(area)
    df["Coltura"]    = crop
    return df

# ─ Aggregazioni
def _add_common_columns(g: pd.DataFrame) -> pd.DataFrame:
    def rnd(col): return 2 if col == "pH medio" else 1
    for col in ["Temp media (°C)","Umidità media (%)","Pioggia tot (mm)","pH medio"]:
        if col in g: g[col] = g[col].round(rnd(col))
    for col in ["Temp min (°C)","Temp max (°C)"]:
        if col in g: g[col] = g[col].round(1)
    for col in ["Giorni raccolta","Giorni piovosi","Quantità tot (q)"]:
        if col in g: g[col] = g[col].astype(int)
    return g

def aggregati_mensili(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Mese", as_index=False).agg(
        **{"Temp media (°C)": ("Temperatura","mean"),
           "Temp min (°C)":   ("Temperatura","min"),
           "Temp max (°C)":   ("Temperatura","max"),
           "Umidità media (%)": ("Umidità","mean"),
           "Pioggia tot (mm)":  ("Precipitazioni","sum"),
           "pH medio":          ("pH","mean"),
           "Quantità tot (q)":  ("Produzione","sum"),
           "Giorni raccolta":   ("Produzione", lambda s: int((s>0).sum())),
           "Giorni piovosi":    ("Precipitazioni", lambda s: int((s>0).sum()))}
    )
    return _add_common_columns(g)

def aggregati_bimestrali(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy(); tmp["Bimestre"] = ((tmp["Mese"] - 1)//2) + 1
    g = tmp.groupby("Bimestre", as_index=False).agg(
        **{"Temp media (°C)": ("Temperatura","mean"),
           "Umidità media (%)": ("Umidità","mean"),
           "Pioggia tot (mm)": ("Precipitazioni","sum"),
           "pH medio": ("pH","mean"),
           "Quantità tot (q)": ("Produzione","sum")}
    )
    return _add_common_columns(g)

def aggregati_semestrali(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy(); tmp["Semestre"] = np.where(tmp["Mese"] <= 6, "H1 (Gen–Giu)", "H2 (Lug–Dic)")
    g = tmp.groupby("Semestre", as_index=False).agg(
        **{"Temp media (°C)": ("Temperatura","mean"),
           "Umidità media (%)": ("Umidità","mean"),
           "Pioggia tot (mm)": ("Precipitazioni","sum"),
           "pH medio": ("pH","mean"),
           "Quantità tot (q)": ("Produzione","sum")}
    )
    g["Semestre"] = pd.Categorical(g["Semestre"], ["H1 (Gen–Giu)","H2 (Lug–Dic)"], ordered=True)
    return _add_common_columns(g.sort_values("Semestre"))

def aggregato_annuale(df: pd.DataFrame, anno: int, coltura: str, area_ha: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([{"Anno": anno,"Coltura": coltura,"Area (ha)": area_ha,"Temp media (°C)": None,
                              "Temp min (°C)": None,"Temp max (°C)": None,"Umidità media (%)": None,
                              "Pioggia tot (mm)": None,"pH medio": None,"Giorni piovosi": 0,
                              "Quantità tot (q)": 0,"Resa media (q/ha)": None,"Giorni raccolta": 0}])
    q_tot = int(df["Produzione"].sum())
    return pd.DataFrame([{
        "Anno": anno, "Coltura": coltura, "Area (ha)": area_ha,
        "Temp media (°C)": round(df["Temperatura"].mean(),1),
        "Temp min (°C)": round(df["Temperatura"].min(),1),
        "Temp max (°C)": round(df["Temperatura"].max(),1),
        "Umidità media (%)": round(df["Umidità"].mean(),1),
        "Pioggia tot (mm)": round(df["Precipitazioni"].sum(),1),
        "pH medio": round(df["pH"].mean(),2),
        "Giorni piovosi": int((df["Precipitazioni"]>0).sum()),
        "Quantità tot (q)": q_tot,
        "Resa media (q/ha)": round(q_tot / max(area_ha,1),1),
        "Giorni raccolta": int((df["Produzione"]>0).sum())
    }])

# ───────────────────────── UI helpers
def kpi_card(title, value):
    return html.Div(className="kpi-card", children=[html.Div(title, className="kpi-title"), html.Div(value, className="kpi-value")])

def get_or_build_year_df(year: int, coltura: str, cache: dict, current_view: dict | None):
    """Restituisce (df, area_ha) per anno+coltura, usando la view corrente se combacia."""
    if current_view and int(current_view.get("year")) == int(year) and current_view.get("coltura") == coltura:
        df_y = pd.DataFrame(current_view["df"])
        area_y = int(current_view.get("area_ha", AREA_TOTALE_HA))
        return df_y, area_y
    mk = f"meteo|{year}"
    meteo_df = pd.DataFrame((cache or {}).get(mk, {}).get("df", []))
    if meteo_df.empty: meteo_df = genera_meteo_anno(year)
    else: meteo_df["Data"] = pd.to_datetime(meteo_df["Data"])
    sk = f"specs|{year}"
    specs = (cache or {}).get(sk, {}).get("specs", None) or genera_specs_colture(year, CROPS, AREA_TOTALE_HA)
    df_y = genera_df_anno_coltura(year, coltura, meteo_df, specs)
    area_y = int(df_y["Area_ha"].iloc[0]) if not df_y.empty else 0
    return df_y, area_y

def summary_2020_to_selected(selected_year: int, coltura: str, cache: dict, current_view: dict | None):
    yrs = [y for y in YEARS if y <= selected_year]  # 2020..anno selezionato
    rows = []
    for y in yrs:
        df_y, area_y = get_or_build_year_df(y, coltura, cache, current_view)
        rows.append(aggregato_annuale(df_y, y, coltura, area_y).iloc[0].to_dict())
    return pd.DataFrame(rows)

# ───────────────────────── App
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = f"{COMPANY_NAME} — Dashboard Produzione"
server = app.server

app.layout = html.Div(children=[
    dcc.Store(id="cache-store", data={}),
    dcc.Store(id="view-store",  data=None),

    html.Div(className="navbar", children=[
        html.Div(className="inner", children=[
            html.Img(src="/assets/logo_orogel.svg", className="logo logo--left",  alt="Logo sinistra"),
            html.Div(className="brand-wrap", children=[
                html.Div(f"{COMPANY_NAME} — Dashboard Produzione", className="brand"),
                html.Div("Riepilogo globale", className="brand-subtitle"),
            ]),
            html.Img(src="/assets/logo_orogel.svg", className="logo logo--right", alt="Logo destra"),
        ])
    ]),

    html.Div(className="container", children=[
        html.H1("Riepilogo globale", className="section"),
        html.Div(className="card", children=[
            html.Div(className="kpi-row", children=[
                html.Div(className="kpi-card", children=[
                    html.Div("Anno", className="kpi-title"),
                    dcc.Dropdown(id="year", options=[{"label": str(y), "value": y} for y in YEARS], value=DEFAULT_YEAR, clearable=False),
                ]),
                html.Div(className="kpi-card", children=[
                    html.Div("Coltura", className="kpi-title"),
                    dcc.Dropdown(id="coltura", options=[{"label": c, "value": c} for c in CROPS], value=DEFAULT_CROP, clearable=False),
                ]),
            ]),
        ]),
        html.Div(id="kpi-row", className="kpi-row"),
        html.Div(className="card", children=[dcc.Tabs(id="tabs", value="tab-1", className="tabs"), html.Div(id="tab-content", className="tab-panel")]),
        html.Div(className="footer", id="footer-info")
    ])
])

# Tabs dinamiche
@app.callback(Output("tabs", "children"), Input("year", "value"), Input("coltura", "value"))
def update_tabs(year, coltura):
    return [
        dcc.Tab(label="Panoramica globale",            value="tab-1", className="tab", selected_className="tab tab--active"),
        dcc.Tab(label="Andamento giornaliero",         value="tab-2", className="tab", selected_className="tab tab--active"),
        dcc.Tab(label="Aggregati (Quantità totale)",   value="tab-3", className="tab", selected_className="tab tab--active"),
        dcc.Tab(label="Correlazioni (mensili)",        value="tab-4", className="tab", selected_className="tab tab--active"),
        dcc.Tab(label="Storico (anni)",                value="tab-6", className="tab", selected_className="tab tab--active"),
        dcc.Tab(label=f"Tabelle & Export (Anno {year} — {coltura})",
                value="tab-5", className="tab", selected_className="tab tab--active"),
    ]

# Cache → View
def _compute_view(df: pd.DataFrame, year: int, coltura: str, gen_ts: str):
    return {"df": df.to_dict("records"),
            "mens": aggregati_mensili(df).to_dict("records"),
            "bims": aggregati_bimestrali(df).to_dict("records"),
            "sems": aggregati_semestrali(df).to_dict("records"),
            "year": int(year), "coltura": coltura,
            "area_ha": int(df["Area_ha"].iloc[0]) if not df.empty else 0,
            "generated_at": gen_ts}

@app.callback(
    Output("view-store", "data"),
    Output("cache-store", "data"),
    Input("year", "value"), Input("coltura", "value"),
    State("cache-store", "data"),
)
def update_view_store(year, coltura, cache):
    cache = cache or {}
    meteo_key = f"meteo|{year}"
    if meteo_key in cache:
        meteo_df = pd.DataFrame(cache[meteo_key]["df"]); meteo_df["Data"] = pd.to_datetime(meteo_df["Data"])
    else:
        meteo_df = genera_meteo_anno(year)
        cache[meteo_key] = {"df": meteo_df.to_dict("records"), "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M")}
    specs_key = f"specs|{year}"
    if specs_key in cache:
        specs = cache[specs_key]["specs"]
    else:
        specs = genera_specs_colture(year, CROPS, AREA_TOTALE_HA)
        cache[specs_key] = {"specs": specs, "generated_at": datetime.now().strftime("%d/%m/%Y %H:%M")}
    key = f"{year}|{coltura}"
    if key in cache:
        df = pd.DataFrame(cache[key]["df"]); gen_ts = cache[key]["generated_at"]
    else:
        df = genera_df_anno_coltura(year, coltura, meteo_df, specs)
        gen_ts = datetime.now().strftime("%d/%m/%Y %H:%M")
        cache[key] = {"df": df.to_dict("records"), "generated_at": gen_ts}
    return _compute_view(df, year, coltura, gen_ts), cache

# KPI + footer
@app.callback(Output("kpi-row", "children"), Output("footer-info", "children"), Input("view-store", "data"))
def update_header(view):
    if not view:
        empty = [kpi_card("Quantità totale (q)", "—"), kpi_card("Resa media (q/ha)", "—"),
                 kpi_card("Temp media (°C)", "—"), kpi_card("Pioggia totale (mm)", "—")]
        return empty, ""
    df = pd.DataFrame(view["df"])
    q_tot = int(df["Produzione"].sum())
    area  = int(view.get("area_ha", AREA_TOTALE_HA)) or 1
    kpis  = [kpi_card("Quantità totale (q)", fmt_int(q_tot)),
             kpi_card("Resa media (q/ha)", round(q_tot/area, 1)),
             kpi_card("Temp media (°C)", round(df["Temperatura"].mean(), 1)),
             kpi_card("Pioggia totale (mm)", round(df["Precipitazioni"].sum(), 1))]
    footer = f"Azienda: {COMPANY_NAME} • Anno: {view['year']} • Coltura: {view['coltura']} • Area: {area} ha • Generato: {view['generated_at']}"
    return kpis, footer

# Contenuto tab
@app.callback(Output("tab-content", "children"),
              Input("tabs", "value"), Input("view-store", "data"),
              State("cache-store", "data"))
def render_tabs(tab, view, cache):
    if not view:
        return html.Div("Caricamento dati…", style={"padding": "12px"})
    DF   = pd.DataFrame(view["df"]).sort_values("Data")
    MENS = pd.DataFrame(view["mens"])
    BIMS = pd.DataFrame(view["bims"])
    SEMS = pd.DataFrame(view["sems"])

    # Base mensile per grafici / correlazioni
    dfm = DF.assign(HotDay=(DF["Temperatura"]>30).astype(int),
                    FrostDay=(DF["Temperatura"]<0).astype(int)).groupby("Mese", as_index=False).agg(
        Quantità=("Produzione","sum"), Pioggia=("Precipitazioni","sum"),
        Temp=("Temperatura","mean"), Umi=("Umidità","mean"), pH=("pH","mean"),
        Giorni_gt30=("HotDay","sum"), Giorni_lt0=("FrostDay","sum")
    ).sort_values("Mese")
    mesi_nomi = [MONTHS_IT[m-1] for m in dfm["Mese"]]

    # Grafici condivisi
    fig_q_month = px.bar(MENS, x="Mese", y="Quantità tot (q)", title="Quantità per mese",
                         labels={"Mese":"Mese","Quantità tot (q)":"q"},
                         color_discrete_sequence=[BAR_COLOR])
    fig_q_month.update_yaxes(range=tight_range(MENS["Quantità tot (q)"]),
                             tickformat=y_tickformat(MENS["Quantità tot (q)"]),
                             title_standoff=14)

    pivot = DF.pivot_table(index="Mese", columns="Giorno", values="Produzione", aggfunc="mean").sort_index()
    fig_heat = go.Figure(go.Heatmap(z=pivot.values, x=[str(c) for c in pivot.columns],
                                    y=[MONTHS_IT[m-1] for m in pivot.index], coloraxis="coloraxis"))
    fig_heat.update_layout(title="Heatmap Quantità: Mese × Giorno (q)", coloraxis=dict(colorscale="Viridis"),
                           height=430, margin=dict(l=40, r=20, t=60, b=20))

    fig_q_rain = make_subplots(specs=[[{"secondary_y": True}]])
    fig_q_rain.add_trace(go.Scatter(x=mesi_nomi, y=dfm["Quantità"], name="Quantità (q)", mode="lines",
                                    line=dict(width=3, color="#4f46e5"), fill="tozeroy",
                                    hovertemplate="%{x}<br>Quantità %{y:,} q<extra></extra>"), secondary_y=False)
    fig_q_rain.add_trace(go.Scatter(x=mesi_nomi, y=dfm["Pioggia"], name="Pioggia (mm)", mode="lines+markers",
                                    line=dict(width=2, color="#22c55e"), marker=dict(size=6),
                                    hovertemplate="%{x}<br>Pioggia %{y:.1f} mm<extra></extra>"), secondary_y=True)
    fig_q_rain.update_yaxes(title_text="Quantità (q)", secondary_y=False,
                            tickformat=y_tickformat(dfm["Quantità"]),
                            range=tight_range(dfm["Quantità"]))
    fig_q_rain.update_yaxes(title_text="Pioggia (mm)", secondary_y=True)
    fig_q_rain.update_layout(title="Quantità mensile vs Pioggia", height=420, margin=dict(l=40,r=20,t=60,b=20))

    fig_q_temp = make_subplots(specs=[[{"secondary_y": True}]])
    fig_q_temp.add_trace(go.Scatter(x=mesi_nomi, y=dfm["Quantità"], name="Quantità (q)", mode="lines",
                                    line=dict(width=3, color="#4f46e5"), fill="tozeroy",
                                    hovertemplate="%{x}<br>Quantità %{y:,} q<extra></extra>"), secondary_y=False)
    fig_q_temp.add_trace(go.Scatter(x=mesi_nomi, y=dfm["Temp"], name="Temp media (°C)", mode="lines+markers",
                                    line=dict(width=2, color="#f59e0b"), marker=dict(size=6),
                                    hovertemplate="%{x}<br>Temp %{y:.1f} °C<extra></extra>"), secondary_y=True)
    fig_q_temp.update_yaxes(title_text="Quantità (q)", secondary_y=False,
                            tickformat=y_tickformat(dfm["Quantità"]),
                            range=tight_range(dfm["Quantità"]))
    fig_q_temp.update_yaxes(title_text="Temp media (°C)", secondary_y=True)
    fig_q_temp.update_layout(title="Quantità mensile vs Temperatura", height=420, margin=dict(l=40,r=20,t=60,b=20))

    total_q = float(dfm["Quantità"].sum()) or 1.0
    share   = (dfm["Quantità"] / total_q * 100.0).round(2)
    fig_ph = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ph.add_trace(go.Bar(x=mesi_nomi, y=share, name="Quota resa annuale (%)", opacity=0.85,
                            marker_color=BAR_COLOR), secondary_y=False)
    fig_ph.add_trace(go.Scatter(x=mesi_nomi, y=dfm["pH"], name="pH (media)", mode="lines+markers", line=dict(width=2)),
                     secondary_y=True)
    fig_ph.update_yaxes(title_text="Quota resa annuale (%)", secondary_y=False)
    fig_ph.update_yaxes(title_text="pH (media)", secondary_y=True, range=[5, 8.5])
    fig_ph.update_layout(title="pH del suolo vs. quota mensile sulla resa annuale (%)", height=420, margin=dict(l=40,r=20,t=60,b=40))

    if tab == "tab-1":  # Panoramica globale
        return html.Div([dcc.Graph(figure=fig_heat),
                         dcc.Graph(figure=fig_q_rain),
                         dcc.Graph(figure=fig_q_temp),
                         dcc.Graph(figure=fig_ph)])

    if tab == "tab-2":  # Andamento giornaliero
        fig_multi = make_subplots(specs=[[{"secondary_y": True}]])
        fig_multi.add_trace(go.Scatter(x=DF["Data"], y=DF["Temperatura"], name="Temp (°C)", mode="lines"), secondary_y=False)
        fig_multi.add_trace(go.Scatter(x=DF["Data"], y=DF["Umidità"],     name="Umidità (%)", mode="lines"), secondary_y=False)
        fig_multi.add_trace(go.Bar    (x=DF["Data"], y=DF["Precipitazioni"], name="Pioggia (mm)", opacity=0.35), secondary_y=True)
        fig_multi.add_trace(go.Scatter(x=DF["Data"], y=DF["Produzione"],  name="Quantità (q)", mode="lines"), secondary_y=True)
        fig_multi.update_yaxes(title_text="Temp (°C) / Umidità (%)", secondary_y=False)
        fig_multi.update_yaxes(title_text="Pioggia (mm) / Quantità (q)", secondary_y=True,
                               tickformat=y_tickformat(DF["Produzione"]))
        fig_multi.update_layout(title="Andamento giornaliero — meteo & quantità", height=520,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                margin=dict(l=40, r=20, t=60, b=20))
        return html.Div([dcc.Graph(figure=fig_multi)])

    if tab == "tab-3":  # Aggregati (Quantità totale)
        b_fig = px.bar(BIMS, x="Bimestre", y="Quantità tot (q)", title="Quantità per bimestre",
                       labels={"Bimestre":"Bimestre","Quantità tot (q)":"q"},
                       color_discrete_sequence=[BAR_COLOR])
        b_fig.update_yaxes(range=tight_range(BIMS["Quantità tot (q)"]),
                           tickformat=y_tickformat(BIMS["Quantità tot (q)"]),
                           title_standoff=14)

        s_fig = px.bar(SEMS, x="Semestre", y="Quantità tot (q)", title="Quantità per semestre",
                       labels={"Semestre":"Semestre","Quantità tot (q)":"q"},
                       color_discrete_sequence=[BAR_COLOR])
        s_fig.update_yaxes(range=tight_range(SEMS["Quantità tot (q)"]),
                           tickformat=y_tickformat(SEMS["Quantità tot (q)"]),
                           title_standoff=14)

        mens_sorted = MENS.sort_values("Mese").assign(**{"Mese nome": lambda d: d["Mese"].map(lambda m: MONTHS_IT[m-1])})
        pie = px.pie(mens_sorted, names="Mese nome", values="Quantità tot (q)",
                     title="Distribuzione percentuale per mese (q)",
                     category_orders={"Mese nome": MONTHS_IT})
        pie.update_traces(textposition="inside", textinfo="percent+label")

        return html.Div([dcc.Graph(figure=fig_q_month, style={"marginBottom":"16px"}),
                         dcc.Graph(figure=b_fig,      style={"marginBottom":"16px"}),
                         dcc.Graph(figure=s_fig,      style={"marginBottom":"16px"}),
                         dcc.Graph(figure=pie)])

    if tab == "tab-4":  # Correlazioni (mensili) con safe_corr
        cols = ["Pioggia","Temp","Umi","pH","Giorni_gt30","Giorni_lt0"]
        labels = {"Pioggia":"Pioggia totale (mm)","Temp":"Temperatura media (°C)","Umi":"Umidità media (%)",
                  "pH":"pH del suolo (media)","Giorni_gt30":"Giorni >30 °C","Giorni_lt0":"Giorni <0 °C"}
        cats    = [labels[c] for c in cols]
        pearson = [safe_corr(dfm["Quantità"], dfm[c]) for c in cols]
        fig_corr = go.Figure(go.Bar(x=cats, y=pearson, name="Pearson (r)", marker_color=BAR_COLOR))
        fig_corr.update_layout(title="Correlazioni (Pearson) con Quantità mensile",
                               yaxis=dict(title="Coefficiente", range=[-1,1], zeroline=True),
                               height=420, margin=dict(l=40,r=20,t=60,b=80), xaxis=dict(tickangle=-20))
        return html.Div([dcc.Graph(figure=fig_corr)])

    if tab == "tab-6":  # Storico (anni)
        coltura = view["coltura"]; cache = cache or {}
        hist = []
        for y in YEARS:
            df_y, _area = get_or_build_year_df(y, coltura, cache, view)
            hist.append({"Anno": y, "Quantità tot (q)": int(df_y["Produzione"].sum())})
        H = pd.DataFrame(hist)
        fig_hist = px.bar(H, x="Anno", y="Quantità tot (q)", title=f"Quantità totale per anno — {coltura}",
                          labels={"Anno":"Anno","Quantità tot (q)":"q"},
                          color_discrete_sequence=[BAR_COLOR])
        fig_hist.update_yaxes(tickformat=y_tickformat(H["Quantità tot (q)"]),
                              range=tight_range(H["Quantità tot (q)"]), title_standoff=14)
        media  = int(H["Quantità tot (q)"].mean())
        best   = H.iloc[H["Quantità tot (q)"].idxmax()]
        worst  = H.iloc[H["Quantità tot (q)"].idxmin()]
        delta  = H.iloc[-1]["Quantità tot (q)"] - H.iloc[0]["Quantità tot (q)"]
        delta_pct = (delta / H.iloc[0]["Quantità tot (q)"] * 100.0) if H.iloc[0]["Quantità tot (q)"] else 0.0
        summary = html.Ul(style={"lineHeight":"1.6"}, children=[
            html.Li(f"Media 2020–2025: {fmt_int(media)} q"),
            html.Li(f"Anno migliore: {int(best['Anno'])} ({fmt_int(int(best['Quantità tot (q)']))} q)"),
            html.Li(f"Anno peggiore: {int(worst['Anno'])} ({fmt_int(int(worst['Quantità tot (q)']))} q)"),
            html.Li(f"Variazione 2020→2025: {fmt_int(int(delta))} q ({delta_pct:+.1f}%)"),
        ])
        return html.Div([dcc.Graph(figure=fig_hist),
                         html.Div(className="card", children=[html.H4("Resoconto 2020–2025", style={"marginTop":0}), summary])])

    # Tabelle & Export — riepilogo dinamico 2020 → anno selezionato
    area = int(view.get("area_ha", AREA_TOTALE_HA))
    annual_df = aggregato_annuale(DF, view["year"], view["coltura"], area)
    RANGEY = summary_2020_to_selected(view["year"], view["coltura"], cache or {}, view)

    st_table = {"overflowX":"auto","border":"1px solid #e5e7eb","borderRadius":"12px"}
    sc_cell  = {"padding":"10px","textAlign":"center","fontFamily":"Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif","fontSize":"14px"}
    sh_head  = {"backgroundColor":"#eef2ff","color":"#1f2937","fontWeight":"700","border":"1px solid #e5e7eb"}
    zebra    = [{"if":{"row_index":"odd"},"backgroundColor":"#fafafa"},
                {"if":{"state":"active"},"backgroundColor":"#f1f5ff","border":"1px solid #c7d2fe"},
                {"if":{"state":"selected"},"backgroundColor":"#e0e7ff"}]

    def mk_table(df):
        return DataTable(data=df.to_dict("records"),
                         columns=[{"name": c, "id": c} for c in df.columns],
                         style_table=st_table, style_cell=sc_cell, style_header=sh_head,
                         style_data_conditional=zebra, page_size=max(1, len(df)))

    def header_row(title, csv_id, dl_id):
        return html.Div(style={"display":"flex","justifyContent":"space-between","alignItems":"center","gap":"12px","marginBottom":"8px"},
                        children=[html.H4(title, style={"margin":0}),
                                  html.Div([html.Button("⬇ Scarica CSV", id=csv_id, n_clicks=0, className="btn btn--primary"),
                                            dcc.Download(id=dl_id)])])

    return html.Div(children=[
        html.Div(className="card", children=[
            header_row(f"Riepilogo 2020–{view['year']}", "btn-5yrs-csv", "dl-5yrs-csv"),
            mk_table(RANGEY)
        ]),
        html.Div(className="card", children=[
            header_row("Riepilogo Annuale", "btn-annual-csv", "dl-annual-csv"),
            mk_table(annual_df)
        ]),
        html.Div(className="card", children=[
            header_row("Riepilogo Mensile", "btn-mens-csv", "dl-mens-csv"),
            mk_table(MENS)
        ]),
        html.Div(className="card", children=[
            header_row("Riepilogo Bimestrale", "btn-bims-csv", "dl-bims-csv"),
            mk_table(BIMS)
        ]),
        html.Div(className="card", children=[
            header_row("Riepilogo Semestrale", "btn-sems-csv", "dl-sems-csv"),
            mk_table(SEMS)
        ]),
    ])

# Download CSV
@app.callback(Output("dl-annual-csv","data"), Input("btn-annual-csv","n_clicks"), State("view-store","data"), prevent_initial_call=True)
def download_annual_csv(n, view):
    if not view: return dash.no_update
    df = pd.DataFrame(view["df"]); area = int(view.get("area_ha", AREA_TOTALE_HA))
    annual = aggregato_annuale(df, view["year"], view["coltura"], area)
    return dcc.send_data_frame(annual.to_csv, f"riepilogo_annuale_{view['year']}_{view['coltura'].replace(' ','_')}.csv", index=False)

@app.callback(Output("dl-mens-csv","data"), Input("btn-mens-csv","n_clicks"), State("view-store","data"), prevent_initial_call=True)
def download_mens_csv(n, view):
    if not view: return dash.no_update
    mens = pd.DataFrame(view["mens"])
    return dcc.send_data_frame(mens.to_csv, f"riepilogo_mensile_{view['year']}_{view['coltura'].replace(' ','_')}.csv", index=False)

@app.callback(Output("dl-bims-csv","data"), Input("btn-bims-csv","n_clicks"), State("view-store","data"), prevent_initial_call=True)
def download_bims_csv(n, view):
    if not view: return dash.no_update
    bims = pd.DataFrame(view["bims"])
    return dcc.send_data_frame(bims.to_csv, f"riepilogo_bimestrale_{view['year']}_{view['coltura'].replace(' ','_')}.csv", index=False)

@app.callback(Output("dl-sems-csv","data"), Input("btn-sems-csv","n_clicks"), State("view-store","data"), prevent_initial_call=True)
def download_sems_csv(n, view):
    if not view: return dash.no_update
    sems = pd.DataFrame(view["sems"])
    return dcc.send_data_frame(sems.to_csv, f"riepilogo_semestrale_{view['year']}_{view['coltura'].replace(' ','_')}.csv", index=False)

# CSV: Riepilogo dinamico 2020 → anno selezionato
@app.callback(
    Output("dl-5yrs-csv","data"),
    Input("btn-5yrs-csv","n_clicks"),
    State("view-store","data"),
    State("cache-store","data"),
    prevent_initial_call=True
)
def download_span_csv(n, view, cache):
    if not view: return dash.no_update
    RANGEY = summary_2020_to_selected(view["year"], view["coltura"], cache or {}, view)
    fname = f"riepilogo_2020_{view['year']}_{view['coltura'].replace(' ','_')}.csv"
    return dcc.send_data_frame(RANGEY.to_csv, fname, index=False)

# Avvio
if __name__ == "__main__":
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:8050", new=2)).start()
    app.run(debug=False)
