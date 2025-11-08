import os, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ==== Utils có sẵn trong repo ====
from utils.data_cleaning import clean_and_log_transform
from utils.feature_engineering import preprocess_and_create_features
from utils.feature_selection import select_features_for_model
from utils.model_scoring import load_lgbm_model, model_feature_names, explain_shap
from utils.policy import load_thresholds, thresholds_for_sector, classify_pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Page config & styles ----------
st.set_page_config(page_title="Corporate Default Risk Scoring", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 0.8rem; padding-bottom: 1.2rem;}
h1,h2,h3 {font-weight: 650;}
.small {font-size:12px; color:#6b7280;}
.metric-card {background:#F8FAFC;border:1px solid #E5E7EB;border-radius:10px;padding:10px 12px;margin-bottom:8px;}
hr {margin: 0.6rem 0;}
</style>
""", unsafe_allow_html=True)

# ---------- Small helpers ----------
ID_LABEL_COLS = {"Year","Ticker","Sector","Exchange","Default"}

def read_csv_smart(path: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            if df.shape[1] == 0:
                raise ValueError("CSV has no columns (empty or bad delimiter).")
            return df
        except Exception:
            continue
    raise RuntimeError(f"Unable to read {path} with common encodings.")

def to_float(x):
    try:
        if pd.isna(x): return np.nan
        if isinstance(x, str): x = x.replace(",", "")
        return float(x)
    except Exception:
        return np.nan

def fmt_money(x):
    return "-" if (x is None or not np.isfinite(x)) else f"{x:,.2f}"

def fmt_ratio(x):
    if (x is None) or (not np.isfinite(x)): return "-"
    return f"{x:.2%}" if -1.5 <= float(x) <= 1.5 else f"{x:,.4f}"

def safe_df(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def force_numeric(X: pd.DataFrame) -> pd.DataFrame:
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return safe_df(X)

def model_align_row(row: pd.Series, model, fallbacks: list) -> pd.DataFrame:
    feats = list(model_feature_names(model) or fallbacks)
    data = {f: float(row.get(f, 0.0)) for f in feats}
    return force_numeric(pd.DataFrame([data], columns=feats))

def align_features_to_model(X_df: pd.DataFrame, model):
    model_feats = list(getattr(model, "feature_name_", []) or [])
    if not model_feats:
        return force_numeric(X_df.copy())
    X = X_df.copy()
    for col in model_feats:
        if col not in X.columns:
            X[col] = 0.0
    return force_numeric(X[model_feats])

def load_train_reference():
    for p in ("models/train_reference.parquet", "models/train_reference.csv"):
        if os.path.exists(p):
            try:
                return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
            except Exception:
                pass
    return None

def bucketize_sector(sector_raw: str) -> str:
    s = (sector_raw or "").lower()
    if any(k in s for k in ["real estate","property","construction"]): return "Real Estate"
    if any(k in s for k in ["steel","material","basic res","cement","mining","metal"]): return "Materials"
    if any(k in s for k in ["energy","oil","gas","coal","petro"]): return "Energy"
    if any(k in s for k in ["bank","finance","insurance","securities"]): return "Financials"
    if any(k in s for k in ["software","it","tech","information"]): return "Technology"
    if any(k in s for k in ["utility","power","water","electric"]): return "Utilities"
    if any(k in s for k in ["staple","food","beverage","agri"]): return "Consumer Staples"
    if any(k in s for k in ["retail","consumer","discretionary","apparel","leisure"]): return "Consumer Discretionary"
    if any(k in s for k in ["industrial","manufacturing","machinery"]): return "Industrials"
    if "tele" in s: return "Telecom"
    if any(k in s for k in ["health","pharma","hospital"]): return "Healthcare"
    if any(k in s for k in ["transport","shipping","airline","airport","logistics"]): return "Transportation"
    if any(k in s for k in ["hotel","hospitality","tourism","travel"]): return "Hospitality & Travel"
    if any(k in s for k in ["auto","automobile","motor"]): return "Automotive"
    if any(k in s for k in ["fish","seafood"]): return "Agriculture & Fisheries"
    return "Other"

# Market microstructure risk weight (sàn)
EXCHANGE_INTENSITY = {"UPCOM": 1.25, "HNX": 1.10, "HOSE": 1.00, "HSX": 1.00}

# ---------- Monte Carlo CVaR ----------
def shrink_cov(cov: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    d = np.diag(np.diag(cov))
    shrunk = (1 - alpha) * cov + alpha * d
    w, V = np.linalg.eigh(shrunk)
    w = np.clip(w, 1e-8, None)
    return (V * w) @ V.T

def mc_cvar_pd(model, Xrow: pd.DataFrame, ref_df: pd.DataFrame,
               sims: int = 4000, alpha: float = 0.95, clip_q=(0.01,0.99)) -> dict:
    assert Xrow.shape[0] == 1
    cols = list(Xrow.columns)
    ref = ref_df[cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    base = Xrow[cols].values.reshape(1,-1).astype(float)[0]
    cov = np.cov(ref.values.T)
    if not np.all(np.isfinite(cov)): cov = np.nan_to_num(cov, nan=0.0)
    cov = shrink_cov(cov, alpha=0.15)
    sims_mat = np.random.multivariate_normal(mean=base, cov=cov, size=sims)
    ql = ref.quantile(clip_q[0], numeric_only=True).values
    qh = ref.quantile(clip_q[1], numeric_only=True).values
    sims_mat = np.minimum(np.maximum(sims_mat, ql), qh)
    X = pd.DataFrame(sims_mat, columns=cols)
    X = align_features_to_model(force_numeric(X), model)
    if hasattr(model, "predict_proba"):
        pd_sims = model.predict_proba(X)[:,1]
    else:
        pd_sims = model.predict(X).astype(float)
    var = float(np.quantile(pd_sims, alpha))
    cvar = float(pd_sims[pd_sims >= var].mean()) if (pd_sims >= var).any() else var
    return {"PD_sims": pd_sims, "VaR": var, "CVaR": cvar}

# ---------- Plotly wrapper: NO deprecated kwargs ----------
def show_plotly(fig, key: str):
    st.plotly_chart(fig, key=key, config={"displayModeBar": False})

# ---------- Load data & model ----------
@st.cache_data(show_spinner=False)
def load_raw_and_features():
    if not os.path.exists("bctc_final.csv"):
        raise FileNotFoundError("bctc_final.csv not found in repository root.")
    raw = read_csv_smart("bctc_final.csv")
    cleaned = clean_and_log_transform(raw.copy())
    feats = preprocess_and_create_features(cleaned)
    return raw, feats

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_lgbm_model("models/lgbm_model.pkl")
    thresholds = load_thresholds("models/threshold.json")
    return model, thresholds

# ---------- Header ----------
st.title("Corporate Default Risk Scoring")

"""
This application has been enhanced to include two additional views alongside the
original probability of default (PD) scoring dashboard.  A set of
buttons at the top of the page allows you to switch between the **Finance**,
**Sentiment** and **Summary** tabs.  The Finance tab provides time–series
views of the selected company’s financial statements and ratios.  The
Sentiment tab displays basic sentiment and news indicators.  The Summary tab
retains the original PD scoring interface and adds a risk commentary section.
"""

# ---------- Data init ----------
try:
    raw_df, feats_df = load_raw_and_features()
except Exception as e:
    st.error(f"Dataset error: {e}")
    st.stop()

try:
    model, thresholds = load_artifacts()
except Exception as e:
    st.error(f"Artifacts error: {e}")
    st.stop()

numeric_cols = [c for c in feats_df.columns if pd.api.types.is_numeric_dtype(feats_df[c])]
candidate_features = [c for c in numeric_cols if c not in ID_LABEL_COLS]
model_feats = model_feature_names(model)
final_features = select_features_for_model(feats_df, candidate_features, model_feats)

# ---------- Sidebar: ticker/year selection ----------
tickers = sorted(feats_df["Ticker"].astype(str).unique().tolist())
ticker = st.sidebar.selectbox("Ticker", tickers, index=0 if tickers else None, key="sb_ticker")

years_avail = sorted(feats_df.loc[feats_df["Ticker"].astype(str)==ticker, "Year"].dropna().astype(int).unique().tolist())
year_idx = len(years_avail)-1 if years_avail else 0
year = st.sidebar.selectbox("Year", years_avail, index=year_idx, key=f"sb_year_{ticker}")

row_model = feats_df[(feats_df["Ticker"].astype(str)==ticker) & (feats_df["Year"]==year)]
if row_model.empty:
    st.warning("No record for selected Ticker & Year.")
    st.stop()
row_model = row_model.iloc[0]

row_raw = raw_df[(raw_df["Ticker"].astype(str)==ticker) & (raw_df["Year"]==year)]
row_raw = row_raw.iloc[0] if not row_raw.empty else pd.Series(dtype="object")

sector_raw = str(row_model.get("Sector","")) if pd.notna(row_model.get("Sector","")) else ""
sector_bucket = bucketize_sector(sector_raw)
exchange = (str(row_model.get("Exchange","")) or "").upper()

# ---------- Session state for tab navigation ----------
if "report_tab" not in st.session_state:
    # default to Summary view on first load
    st.session_state.report_tab = "Summary"

# Create horizontal buttons for tab switching
btn_col1, btn_col2, btn_col3 = st.columns(3)
with btn_col1:
    if st.button("💹 Finance", key="btn_finance", use_container_width=True,
                 type="primary" if st.session_state.report_tab == "Finance" else "secondary"):
        st.session_state.report_tab = "Finance"
with btn_col2:
    if st.button("📰 Sentiment", key="btn_sentiment", use_container_width=True,
                 type="primary" if st.session_state.report_tab == "Sentiment" else "secondary"):
        st.session_state.report_tab = "Sentiment"
with btn_col3:
    if st.button("📈 Summary", key="btn_summary", use_container_width=True,
                 type="primary" if st.session_state.report_tab == "Summary" else "secondary"):
        st.session_state.report_tab = "Summary"

def get_raw(col_names, default=np.nan):
    for c in col_names:
        if c in row_raw.index:
            return to_float(row_raw[c])
    return default

assets_raw = get_raw(["TOTAL ASSETS (Bn. VND)","Total_Assets"])
equity_raw = get_raw(["OWNER'S EQUITY(Bn.VND)","Equity"])
curr_liab = get_raw(["Current liabilities (Bn. VND)","Current_Liabilities"], 0.0)
long_liab = get_raw(["Long-term liabilities (Bn. VND)","Long_Term_Liabilities"], 0.0)
short_bor = get_raw(["Short-term borrowings (Bn. VND)","Short_Term_Borrowings"], 0.0)

revenue_raw = get_raw(["Net Sales","Revenue"])
net_profit_raw = get_raw(["Net Profit For the Year","Net_Profit"])
oper_profit_raw = get_raw(["Operating Profit/Loss","Operating_Profit"])
interest_exp_raw = get_raw(["Interest Expenses","Interest_Expenses"], 0.0)
cash_raw = get_raw(["Cash and cash equivalents (Bn. VND)","Cash"], 0.0)
receivables_raw = get_raw(["Accounts receivable (Bn. VND)","Receivables"], 0.0)
inventories_raw = get_raw(["Net Inventories","Inventories"], 0.0)
current_assets_raw = get_raw(["CURRENT ASSETS (Bn. VND)","Current_Assets"], 0.0)

def safe_div(a, b):
    try:
        return (float(a) / float(b)) if (b not in [0, None, np.nan] and float(b)!=0.0) else np.nan
    except Exception:
        return np.nan

total_liab_raw = (curr_liab or 0.0) + (long_liab or 0.0)
interest_bearing_debt = (short_bor or 0.0) + (long_liab or 0.0)
debt_raw = to_float(row_raw.get("Total_Debt")) if ("Total_Debt" in row_raw.index and pd.notna(row_raw.get("Total_Debt"))) else interest_bearing_debt

roa = safe_div(net_profit_raw, assets_raw)
roe = safe_div(net_profit_raw, equity_raw)
dta = safe_div(total_liab_raw, assets_raw); dta = min(max(dta, 0.0), 0.999) if pd.notna(dta) else np.nan
dte = safe_div(debt_raw, equity_raw);     dte = min(max(dte, 0.0), 0.999) if pd.notna(dte) else np.nan
current_ratio = safe_div(current_assets_raw, curr_liab)
quick_ratio   = safe_div((cash_raw or 0.0) + (receivables_raw or 0.0), curr_liab)

# ---------- Sidebar metrics ----------
with st.sidebar:
    st.header("Company Profile")
    st.subheader(f"{ticker} — {int(year)}")
    st.markdown(f"**Sector:** {sector_raw or '-'}  \n**Exchange:** {exchange or '-'}")
    # display key numeric figures in a card
    st.markdown(
        "<div class='metric-card'>"
        f"Total Assets: <b>{fmt_money(assets_raw)}</b><br>"
        f"Equity: <b>{fmt_money(equity_raw)}</b><br>"
        f"Debt: <b>{fmt_money(debt_raw)}</b><br>"
        f"Revenue: <b>{fmt_money(revenue_raw)}</b><br>"
        f"Net Profit: <b>{fmt_money(net_profit_raw)}</b>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='metric-card'>"
        f"ROA: <b>{fmt_ratio(roa)}</b><br>"
        f"ROE: <b>{fmt_ratio(roe)}</b><br>"
        f"Debt/Equity: <b>{fmt_ratio(dte)}</b><br>"
        f"Debt/Assets: <b>{fmt_ratio(dta)}</b>"
        "</div>",
        unsafe_allow_html=True
    )

# ===========================================================================
#  Financial and Sentiment rendering utilities
#
#  To keep the main body concise, we encapsulate the new Finance and
#  Sentiment views in helper functions.  These functions operate on the
#  pre-loaded `raw_df` dataframe and leverage the selected `ticker` to
#  extract historical series.  Where numeric transformations are needed we
#  reuse helper functions (`to_float`, `safe_div`) defined above.  Each view
#  constructs simple tables and charts using Streamlit primitives.  Should
#  you wish to enrich these sections further (e.g. add more line items or
#  refine the narrative), you can edit the functions below without touching
#  the core PD scoring logic.

def _extract_value(row: pd.Series, cols: list[str]):
    """Helper to extract the first non-null value from a list of columns."""
    for col in cols:
        if col in row and pd.notna(row[col]):
            return to_float(row[col])
    return np.nan

def render_finance_tab():
    """
    Render the Finance tab.
    This view displays simplified Income Statement, Balance Sheet,
    Cashflow Statement, and financial ratios across all available years
    for the selected ticker.  At the bottom, a note summarises key
    trends.  Users may edit this narrative manually later.
    """
    # filter historical rows for the selected ticker
    hist = raw_df[raw_df["Ticker"].astype(str) == ticker].copy()
    if hist.empty:
        st.info("No historical data available for this ticker.")
        return
    hist = hist.sort_values("Year").reset_index(drop=True)

    # ensure year column is int for sorting
    hist["Year"] = pd.to_numeric(hist["Year"], errors="coerce")
    years_sorted = hist["Year"].dropna().astype(int).unique().tolist()

    # Income Statement table
    income_items = [
        ("Revenue", ["Net Sales", "Revenue (Bn. VND)", "Revenue"]),
        ("Gross Profit", ["Gross Profit"]),
        ("Operating Profit", ["Operating Profit/Loss"]),
        ("Profit Before Tax", ["Profit before tax", "Net Profit/Loss before tax"]),
        ("Net Profit", ["Net Profit For the Year"]),
    ]
    income_data = []
    for _, row in hist.iterrows():
        yr = int(row.get("Year")) if pd.notna(row.get("Year")) else None
        row_dict = {"Year": yr}
        for name, cols in income_items:
            row_dict[name] = _extract_value(row, cols)
        income_data.append(row_dict)
    income_df = pd.DataFrame(income_data)
    if not income_df.empty:
        income_df = income_df.set_index("Year").sort_index()
    
    # Balance Sheet table
    bs_items = [
        ("Cash & CE", ["Cash and cash equivalents (Bn. VND)", "Cash and cash equivalents", "Cash"]),
        ("Accounts Receivable", ["Accounts receivable (Bn. VND)", "Short-term loans receivables (Bn. VND)"]),
        ("Inventories", ["Net Inventories", "Inventories, Net (Bn. VND)"]),
        ("Other Current Assets", ["Other current assets", "Other current assets (Bn. VND)"]),
        ("Long-term Assets", ["LONG-TERM ASSETS (Bn. VND)"]),
        ("Fixed Assets", ["Fixed assets (Bn. VND)"]),
        ("Long-term Investments", ["Long-term investments (Bn. VND)"]),
        ("Other Non-current Assets", ["Other non-current assets"]),
        ("Total Assets", ["TOTAL ASSETS (Bn. VND)"]),
        ("Total Liabilities", ["LIABILITIES (Bn. VND)", "Total liabilities (Bn. VND)"]),
        ("Current Liabilities", ["Current liabilities (Bn. VND)"]),
        ("Long-term Liabilities", ["Long-term liabilities (Bn. VND)"]),
        ("Equity", ["OWNER'S EQUITY(Bn.VND)", "Equity"]),
    ]
    bs_data = []
    for _, row in hist.iterrows():
        yr = int(row.get("Year")) if pd.notna(row.get("Year")) else None
        row_dict = {"Year": yr}
        for name, cols in bs_items:
            row_dict[name] = _extract_value(row, cols)
        bs_data.append(row_dict)
    bs_df = pd.DataFrame(bs_data)
    if not bs_df.empty:
        bs_df = bs_df.set_index("Year").sort_index()
    
    # Cashflow Statement table
    cf_items = [
        ("Operating CF", ["Net cash inflows/outflows from operating activities", "Net cash flows from operating activities"]),
        ("Investing CF", ["Net Cash Flows from Investing Activities"]),
        ("Financing CF", ["Cash flows from financial activities"]),
        ("Net Change in Cash", ["Net increase/decrease in cash and cash equivalents"]),
        ("Ending Cash", ["Cash and Cash Equivalents at the end of period", "Cash and cash equivalents"]),
    ]
    cf_data = []
    for _, row in hist.iterrows():
        yr = int(row.get("Year")) if pd.notna(row.get("Year")) else None
        row_dict = {"Year": yr}
        for name, cols in cf_items:
            row_dict[name] = _extract_value(row, cols)
        cf_data.append(row_dict)
    cf_df = pd.DataFrame(cf_data)
    if not cf_df.empty:
        cf_df = cf_df.set_index("Year").sort_index()

    # Financial ratios table
    ratio_rows = []
    for _, row in hist.iterrows():
        yr = int(row.get("Year")) if pd.notna(row.get("Year")) else None
        # extract relevant raw values
        assets  = _extract_value(row, ["TOTAL ASSETS (Bn. VND)", "Total_Assets"])
        equity  = _extract_value(row, ["OWNER'S EQUITY(Bn.VND)", "Equity"])
        curr_liab = _extract_value(row, ["Current liabilities (Bn. VND)"])
        long_liab = _extract_value(row, ["Long-term liabilities (Bn. VND)"])
        debt = _extract_value(row, ["Total_Debt", "Long-term borrowings (Bn. VND)", "Short-term borrowings (Bn. VND)"])
        revenue  = _extract_value(row, ["Net Sales", "Revenue (Bn. VND)", "Revenue"])
        net_profit = _extract_value(row, ["Net Profit For the Year"])
        cash_val = _extract_value(row, ["Cash and cash equivalents (Bn. VND)", "Cash and cash equivalents"])
        receivables_val = _extract_value(row, ["Accounts receivable (Bn. VND)"])
        current_assets = _extract_value(row, ["CURRENT ASSETS (Bn. VND)", "Current_Assets"])
        total_liab = (curr_liab or 0.0) + (long_liab or 0.0)
        # compute ratios
        r_roa = safe_div(net_profit, assets)
        r_roe = safe_div(net_profit, equity)
        r_dta = safe_div(total_liab, assets)
        r_dte = safe_div(debt, equity)
        r_cr  = safe_div(current_assets, curr_liab)
        r_qr  = safe_div((cash_val or 0.0) + (receivables_val or 0.0), curr_liab)
        ratio_rows.append({
            "Year": yr,
            "ROA": r_roa,
            "ROE": r_roe,
            "Debt_to_Assets": r_dta,
            "Debt_to_Equity": r_dte,
            "Current_Ratio": r_cr,
            "Quick_Ratio": r_qr,
        })
    ratio_df = pd.DataFrame(ratio_rows)
    if not ratio_df.empty:
        ratio_df = ratio_df.set_index("Year").sort_index()
    
    # Display sections
    st.subheader("Income Statement (Bn. VND)")
    if income_df.empty:
        st.info("No income statement data found.")
    else:
        # format numbers for display
        fmt_inc = income_df.applymap(lambda x: fmt_money(x) if pd.notna(x) else "-")
        st.dataframe(fmt_inc, use_container_width=True)

    st.subheader("Balance Sheet (Bn. VND)")
    if bs_df.empty:
        st.info("No balance sheet data found.")
    else:
        fmt_bs = bs_df.applymap(lambda x: fmt_money(x) if pd.notna(x) else "-")
        st.dataframe(fmt_bs, use_container_width=True)

    st.subheader("Cash Flow Statement (Bn. VND)")
    if cf_df.empty:
        st.info("No cash flow statement data found.")
    else:
        fmt_cf = cf_df.applymap(lambda x: fmt_money(x) if pd.notna(x) else "-")
        st.dataframe(fmt_cf, use_container_width=True)

    st.subheader("Financial Ratios")
    if ratio_df.empty:
        st.info("No ratio data found.")
    else:
        # pivot so metrics are rows and years are columns for readability
        pivot_ratio = ratio_df.T
        pivot_ratio = pivot_ratio.applymap(lambda x: fmt_ratio(x) if pd.notna(x) else "-")
        st.dataframe(pivot_ratio, use_container_width=True)

    # Generate a simple narrative note based on trends
    try:
        note_lines = []
        # revenue trend
        if not income_df["Revenue"].dropna().empty:
            rev_start = income_df["Revenue"].dropna().iloc[0]
            rev_end   = income_df["Revenue"].dropna().iloc[-1]
            if pd.notna(rev_start) and pd.notna(rev_end):
                if rev_end > rev_start * 1.05:
                    note_lines.append("Doanh thu tăng trưởng ổn định trong giai đoạn gần đây.")
                elif rev_end < rev_start * 0.95:
                    note_lines.append("Doanh thu có xu hướng giảm đáng kể so với những năm trước.")
                else:
                    note_lines.append("Doanh thu dao động nhẹ nhưng không thay đổi nhiều.")
        # profit trend
        if not income_df["Net Profit"].dropna().empty:
            pr_start = income_df["Net Profit"].dropna().iloc[0]
            pr_end   = income_df["Net Profit"].dropna().iloc[-1]
            if pd.notna(pr_start) and pd.notna(pr_end):
                if pr_end > pr_start * 1.05:
                    note_lines.append("Lợi nhuận ròng cải thiện qua thời gian, cho thấy hoạt động hiệu quả hơn.")
                elif pr_end < pr_start * 0.95:
                    note_lines.append("Lợi nhuận ròng suy giảm, cần xem xét nguyên nhân chi phí hoặc doanh thu.")
                else:
                    note_lines.append("Lợi nhuận ròng ổn định qua các năm.")
        # leverage
        if not ratio_df["Debt_to_Equity"].dropna().empty:
            dte_latest = ratio_df["Debt_to_Equity"].dropna().iloc[-1]
            if pd.notna(dte_latest):
                if dte_latest < 1:
                    note_lines.append("Tỷ lệ nợ trên vốn chủ sở hữu ở mức an toàn (<1), cho thấy khả năng tự tài trợ tốt.")
                elif dte_latest < 2:
                    note_lines.append("Tỷ lệ nợ trên vốn chủ sở hữu trung bình, công ty dùng đòn bẩy nợ tương đối.")
                else:
                    note_lines.append("Tỷ lệ nợ trên vốn chủ sở hữu cao, tiềm ẩn rủi ro đòn bẩy tài chính.")
        # liquidity
        if not ratio_df["Current_Ratio"].dropna().empty:
            cr_latest = ratio_df["Current_Ratio"].dropna().iloc[-1]
            if pd.notna(cr_latest):
                if cr_latest > 1.5:
                    note_lines.append("Hệ số thanh khoản hiện hành cao (>1.5), khả năng đáp ứng nợ ngắn hạn tốt.")
                elif cr_latest > 1.0:
                    note_lines.append("Hệ số thanh khoản hiện hành ở mức trung bình, cần theo dõi." )
                else:
                    note_lines.append("Hệ số thanh khoản hiện hành thấp (<1), công ty có thể gặp khó khăn trong thanh toán ngắn hạn.")
    except Exception:
        note_lines = []

    st.subheader("Note / Đánh giá")
    if not note_lines:
        st.write("*Phần ghi chú mẫu sẽ hiển thị ở đây. Bạn có thể cập nhật nội dung theo ý muốn.*")
    else:
        for line in note_lines:
            st.write(f"- {line}")

def render_sentiment_tab():
    """
    Render the Sentiment tab.
    Displays sentiment-related metrics available in the dataset and
    classifies overall market tone.  Since raw news headlines are not
    included, placeholders are provided for users to add their own
    commentary or integrate external feeds in the future.
    """
    hist = raw_df[raw_df["Ticker"].astype(str) == ticker].copy()
    if hist.empty:
        st.info("No historical sentiment data available for this ticker.")
        return
    hist = hist.sort_values("Year")
    hist["Year"] = pd.to_numeric(hist["Year"], errors="coerce")
    # extract sentiment-related columns if they exist
    sen_cols = [c for c in hist.columns if c.lower() in ["sentiment change", "news shock", "sentiment_change", "news_shock"]]
    if not sen_cols:
        st.info("Dataset does not contain sentiment metrics.")
        sen_df = pd.DataFrame()
    else:
        sen_df = hist[["Year"] + sen_cols].copy().dropna()
        sen_df = sen_df.set_index("Year").sort_index()
    
    st.subheader("Sentiment Metrics by Year")
    if sen_df.empty:
        st.info("Không có dữ liệu sentiment cho mã cổ phiếu này.")
    else:
        # classify sentiment direction
        def classify_sent(val):
            try:
                v = float(val)
                if v > 0.01: return "Positive"
                if v < -0.01: return "Negative"
                return "Neutral"
            except Exception:
                return "Neutral"
        df_disp = sen_df.copy()
        for col in sen_cols:
            df_disp[f"{col} Classification"] = df_disp[col].apply(classify_sent)
        st.dataframe(df_disp, use_container_width=True)
        # simple line plot for the first sentiment metric
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            first_col = sen_cols[0]
            ax.plot(df_disp.index.astype(int), df_disp[first_col])
            ax.set_title(f"{first_col} over Time")
            ax.set_xlabel("Year")
            ax.set_ylabel(first_col)
            st.pyplot(fig)
        except Exception:
            pass
    
    st.subheader("News & Market Tone")
    st.write(
        "Hiện tại chưa có dữ liệu tin tức chi tiết. Bạn có thể thêm danh sách tiêu đề "
        "tin tức và nhận định của mình tại đây. Ví dụ: *Công ty công bố kết quả kinh "
        "doanh vượt kỳ vọng* (tích cực) hoặc *CEO bất ngờ từ chức* (tiêu cực)."
    )

# ---------- Tab Routing ----------
# If the current view is not the Summary, call the appropriate renderer and
# halt further execution to avoid mixing content.  This ensures each tab
# displays only its relevant widgets and charts.  The Summary tab falls
# through to the original PD scoring dashboard below.
if st.session_state.report_tab == "Finance":
    render_finance_tab()
    st.stop()
elif st.session_state.report_tab == "Sentiment":
    render_sentiment_tab()
    st.stop()

# ---------- Model input ----------
X_base = model_align_row(row_model, model, fallbacks=final_features)
X_base = align_features_to_model(X_base, model)

# ===================== A) Company Overview =====================
st.subheader("A. Company Financial Overview")

hist = raw_df[raw_df["Ticker"].astype(str)==ticker].sort_values("Year")
rev_series = hist[["Year","Net Sales","Net Profit For the Year"]].rename(
    columns={"Net Sales":"Revenue","Net Profit For the Year":"Net_Profit"}
).dropna(how="any")

col1, col2 = st.columns([2,1])
with col1:
    if not rev_series.empty:
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Bar(x=rev_series["Year"], y=rev_series["Revenue"], name="Revenue"))
        fig_rev.add_trace(go.Scatter(x=rev_series["Year"], y=rev_series["Net_Profit"], name="Net Profit", mode="lines+markers", yaxis="y2"))
        fig_rev.update_layout(
            title="Revenue & Net Profit (multi-year)",
            yaxis=dict(title="Revenue"),
            yaxis2=dict(title="Net Profit", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            height=380
        )
        show_plotly(fig_rev, "ov_rev")
    else:
        st.info("No historical series for this company.")

with col2:
    fig_cap = go.Figure(data=[go.Pie(labels=["Total Debt","Equity"], values=[debt_raw, equity_raw], hole=0.5)])
    fig_cap.update_layout(title="Capital Structure", height=380)
    show_plotly(fig_cap, "ov_cap")

st.markdown("### Key Financial Ratios")
key_ratios = pd.DataFrame({
    "Metric": ["ROA","ROE","Debt_to_Assets","Debt_to_Equity","Current_Ratio","Quick_Ratio"],
    "Value": [roa, roe, dta, dte, current_ratio, quick_ratio]
})
key_ratios["Value"] = key_ratios["Value"].apply(fmt_ratio)
st.dataframe(key_ratios, use_container_width=True, hide_index=True)

# ===================== B) Default Probability (PD) & Policy Band =====================
st.subheader("B. Default Probability (PD) & Policy Band")

def _logit(p, eps=1e-9):
    p = float(np.clip(p, eps, 1 - eps))
    return np.log(p / (1 - p))

def _sigmoid(z):
    z = float(z)
    if z >= 35:  return 1.0
    if z <= -35: return 0.0
    return 1.0 / (1.0 + np.exp(-z))

# --- NGƯỠNG CHUẨN THEO YÊU CẦU ---
LOW_CUT  = 0.20   # Low < 20%
MED_CUT  = 0.50   # 20% <= Medium < 50%
# ---------------------------------

# 1) PD từ model
pd_model = float(model.predict_proba(X_base)[:, 1][0]) if hasattr(model, "predict_proba") \
           else float(model.predict(X_base)[0])

# 2) Per-ticker overrides (giữ như bạn có)
TICKER_OVERRIDES = {
    "HAG": {"logit_boost": 2.20, "severity_boost": 0.50, "pd_floor": 0.45},
    "ROS": {"logit_boost": 1.60, "severity_boost": 0.40, "pd_floor": 0.30},
}

# 3) Cấu hình hậu-hiệu chỉnh (giữ như bạn có)
PD_CFG = {
    "exchange_logit_mult": {"UPCOM": 1.10, "HNX": 0.45, "HOSE": 0.00, "HSX": 0.00, "__default__": 0.20},
    "size": {"assets_q40": 0.35, "revenue_q40": 0.20},
    "leverage": {"dta_hi": 0.50, "dte_hi": 0.40, "netde_hi": 0.35},
    "profitability": {"roa_neg": 0.50, "roe_neg": 0.35, "npm_neg": 0.30, "rev_cagr_neg": 0.25},
    "liquidity": {"cr_low": 0.25, "qr_low": 0.20},
    "governance": {"auditor_non_big4": 0.25, "opinion_qualified": 0.70, "filing_delay": 0.25},
    "sector_tilt": {
        "Real Estate": 0.60, "Materials": 0.25, "Consumer Discretionary": 0.15,
        "Financials": 0.00, "Utilities": -0.05, "Technology": 0.00, "__default__": 0.05
    },
    "pd_floor": {"UPCOM": 0.15, "HNX": 0.08, "HOSE": 0.03, "HSX": 0.03, "__default__": 0.05},
    "pd_cap": {"default": 0.98}
}

# 4) Lấy tín hiệu rủi ro từ row (giữ nguyên)
def _get(sr, keys, default=np.nan):
    for k in keys:
        if k in sr.index and pd.notna(sr.get(k)):
            try: return float(sr.get(k))
            except Exception: return default
    return default

npm = _get(row_model, ["Net_Profit_Margin","net_profit_margin"])
rev_cagr3y = _get(row_model, ["Revenue_CAGR_3Y","revenue_cagr_3y","sales_cagr_3y"])
nde = _get(row_model, ["Net_Debt_to_Equity","net_debt_to_equity"])
auditor = str(_get(row_raw, ["Auditor","Audit_Firm","Auditor_Name"], "") or "")
opinion = str(_get(row_raw, ["Audit_Opinion","Opinion"], "") or "")
filing_delay = _get(row_raw, ["Filing_Delay_Days","Filing_Delay"], np.nan)

# 5) Phân vị size (giữ nguyên)
ref = load_train_reference(); ref_use = ref if isinstance(ref,pd.DataFrame) else feats_df
def _q(col, q, fallback=np.nan):
    if (col in ref_use.columns) and ref_use[col].notna().any():
        try: return float(pd.to_numeric(ref_use[col], errors="coerce").quantile(q))
        except Exception: return fallback
    return fallback
assets_q40  = _q("Total_Assets", 0.40, np.nan) if "Total_Assets" in ref_use.columns else np.nan
revenue_q40 = _q("Revenue",      0.40, np.nan) if "Revenue"      in ref_use.columns else np.nan

# 6) Cờ rủi ro (giữ nguyên)
flags = {
    "exch_mult": PD_CFG["exchange_logit_mult"].get(exchange, PD_CFG["exchange_logit_mult"]["__default__"]),
    "assets_q40": (np.isfinite(assets_raw) and np.isfinite(assets_q40) and assets_raw < assets_q40),
    "revenue_q40": (np.isfinite(revenue_raw) and np.isfinite(revenue_q40) and revenue_raw < revenue_q40),
    "dta_hi": (isinstance(dta, float) and dta > 0.70),
    "dte_hi": (isinstance(dte, float) and dte > 1.5),
    "netde_hi": (isinstance(nde, float) and nde > 1.0),
    "roa_neg": (isinstance(roa, float) and roa < 0.0),
    "roe_neg": (isinstance(roe, float) and roe < 0.0),
    "npm_neg": (isinstance(npm, float) and npm < 0.0),
    "rev_cagr_neg": (isinstance(rev_cagr3y, float) and rev_cagr3y < 0.0),
    "cr_low": (isinstance(current_ratio, float) and current_ratio < 0.9),
    "qr_low": (isinstance(quick_ratio, float) and quick_ratio < 0.7),
    "auditor_non_big4": (auditor != "" and not any(k in auditor.lower() for k in ["deloitte","kpmg","ey","ernst","pwc","pricewaterhouse"])),
    "opinion_qualified": (opinion != "" and any(k in opinion.lower() for k in ["qualified","adverse","disclaimer"])),
    "filing_delay": (isinstance(filing_delay, float) and filing_delay >= 20),
}

# 7) Risk intensity (giữ nguyên)
risk_intensity = 1.0
for cond, bump in [
    ("dta_hi",0.25), ("dte_hi",0.20), ("netde_hi",0.15),
    ("cr_low",0.15), ("qr_low",0.10),
    ("roa_neg",0.20), ("roe_neg",0.10), ("npm_neg",0.10), ("rev_cagr_neg",0.10),
    ("assets_q40",0.10), ("revenue_q40",0.05)
]:
    if flags[cond]: risk_intensity += bump
if exchange == "UPCOM": risk_intensity += 0.25
risk_intensity = float(np.clip(risk_intensity, 1.0, 2.5))

# 8) Hậu-hiệu chỉnh logit (giữ nguyên)
logit0 = _logit(pd_model)
adj = 0.0
adj += flags["exch_mult"]
adj += PD_CFG["sector_tilt"].get(sector_bucket, PD_CFG["sector_tilt"]["__default__"])
for group_cfg, conds in [
    (PD_CFG["size"], ["assets_q40","revenue_q40"]),
    (PD_CFG["leverage"], ["dta_hi","dte_hi","netde_hi"]),
    (PD_CFG["profitability"], ["roa_neg","roe_neg","npm_neg","rev_cagr_neg"]),
    (PD_CFG["liquidity"], ["cr_low","qr_low"]),
    (PD_CFG["governance"], ["auditor_non_big4","opinion_qualified","filing_delay"]),
]:
    for c in conds:
        if flags[c]: adj += group_cfg[c]

ovr = TICKER_OVERRIDES.get(str(ticker), {})
adj += float(ovr.get("logit_boost", 0.0))
risk_intensity += float(ovr.get("risk_boost", 0.0))
adj *= risk_intensity

pd_floor = float(ovr.get("pd_floor", PD_CFG["pd_floor"].get(exchange, PD_CFG["pd_floor"]["__default__"])))
pd_cap   = PD_CFG["pd_cap"]["default"]
pd_final = float(np.clip(_sigmoid(logit0 + adj), pd_floor, pd_cap))

# 9) Phân loại band theo ngưỡng cố định 20% / 50%
def policy_band_fixed(pd_val: float) -> str:
    pv = float(pd_val)
    if pv < LOW_CUT: return "Low"
    if pv < MED_CUT: return "Medium"
    return "High"

band = policy_band_fixed(pd_final)

# 10) Render
c1, c2, c3 = st.columns([1, 1, 2])
with c1: st.metric("PD (multi-factor, post-adj.)", f"{pd_final:.2%}")
with c2: st.metric("Policy Band", band)
with c3:
    st.markdown(
        f"""
        <div class='small'>
          <span style="display:inline-flex;align-items:center;gap:8px;">
            <span style="display:inline-block;width:14px;height:14px;background:#E8F1FB;border:1px solid #cbd5e1;border-radius:3px;"></span>
            Low &lt; {LOW_CUT:.0%}
            <span style="display:inline-block;width:14px;height:14px;background:#CFE3F7;border:1px solid #cbd5e1;border-radius:3px;margin-left:16px;"></span>
            Medium &lt; {MED_CUT:.0%}
            <span style="display:inline-block;width:14px;height:14px;background:#F9E3E3;border:1px solid #cbd5e1;border-radius:3px;margin-left:16px;"></span>
            High ≥ {MED_CUT:.0%} • Floor/Cap: {pd_floor:.0%}/{pd_cap:.0%} • Exchange: {exchange or '-'}
          </span>
        </div>
        """,
        unsafe_allow_html=True
    )

gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=pd_final * 100,
    number={'suffix': "%"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': '#1f77b4'},
        'steps': [
            {'range': [0, LOW_CUT * 100],          'color': '#E8F1FB'},  # Low
            {'range': [LOW_CUT * 100, MED_CUT*100],'color': '#CFE3F7'},  # Medium
            {'range': [MED_CUT * 100, 100],        'color': '#F9E3E3'},  # High
        ],
        'threshold': {'line': {'color': 'red', 'width': 3}, 'value': pd_final * 100}
    }
))
gauge.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
show_plotly(gauge, "pd_gauge")

# ===================== C) SHAP (gọn, chắc) =====================
st.subheader("C. Model Explainability (SHAP)")
fig_sh = None
try:
    shap_raw = explain_shap(model, X_base, top_n=10)
    if shap_raw is None:
        shap_df = pd.DataFrame()
    elif isinstance(shap_raw, pd.Series):
        shap_df = shap_raw.reset_index(); shap_df.columns = ["Feature","SHAP"]
    elif isinstance(shap_raw, (list, tuple, np.ndarray)):
        try: shap_df = pd.DataFrame(shap_raw, columns=["Feature","SHAP"])
        except Exception: shap_df = pd.DataFrame()
    elif isinstance(shap_raw, pd.DataFrame):
        shap_df = shap_raw.copy()
    else:
        shap_df = pd.DataFrame()
except Exception:
    shap_df = pd.DataFrame()

def _pick_col(df: pd.DataFrame, cands):
    lower = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in df.columns: return c
        if c.lower() in lower: return lower[c.lower()]
    return None

if shap_df.empty:
    st.info("SHAP is not available for this model/input.")
else:
    feat_col = _pick_col(shap_df, ["Feature","feature","name","variable"])
    shap_col = _pick_col(shap_df, ["SHAP","shap","impact","value","shap_value"])
    if (feat_col is None) or (shap_col is None):
        st.info("SHAP output detected but columns are not recognizable.")
    else:
        shap_df = shap_df[[feat_col, shap_col]].dropna()
        shap_df[shap_col] = pd.to_numeric(shap_df[shap_col], errors="coerce")
        shap_df = shap_df.dropna()
        if not shap_df.empty:
            shap_df["absSHAP"] = shap_df[shap_col].abs()
            shap_df = shap_df.sort_values("absSHAP", ascending=True).tail(10)
            fig_sh = go.Figure()
            colors = ["#E24A33" if v < 0 else "#1F77B4" for v in shap_df[shap_col]]
            fig_sh.add_trace(go.Bar(
                x=shap_df[shap_col], y=shap_df[feat_col].astype(str),
                orientation="h", marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_df[shap_col]], textposition="outside"
            ))
            fig_sh.update_layout(title="Top Feature Contributions (SHAP)",
                                 xaxis=dict(title="SHAP value → PD"),
                                 height=420, margin=dict(l=10, r=20, t=40, b=10))
            show_plotly(fig_sh, "shap_top")

# ===================== D) Stress Testing — Sector & Systemic (Preset, no calc) =====================
st.subheader("D. Stress Testing — Sector & Systemic Impacts")

# 1) Danh mục kịch bản hiển thị theo nhóm
SECTOR_SCENARIOS = {
    "Real Estate": ["Credit Tightening", "Property Price Correction"],
    "Materials": ["Steel Price Collapse", "Energy Cost Surge"],
    "Energy": ["Oil Demand Crash", "Field Outage"],
    "Technology": ["Valuation Reset", "Supply Chain Disruptions"],
    "Consumer Discretionary": ["COVID Demand Shock", "Luxury Slowdown"],
    "Consumer Staples": ["Energy Price Shock", "Food Input Spike"],
    "Industrials": ["Logistics/Supply Chain", "Export Order Drop"],
    "Utilities": ["Regulatory Tightening"],
    "Financials": ["Credit Loss Cycle", "Funding Cost Rise"],
    "Healthcare": ["Reimbursement Pressure"],
    "Telecom": ["Capex Cycle Upswing"],
    "Transportation": ["Travel Collapse", "Fuel Spike"],
    "Hospitality & Travel": ["Tourism Freeze"],
    "Agriculture & Fisheries": ["Export Shock"],
    "Automotive": ["Semiconductor Shortage"],
    "Other": ["Generic Sector Shock"]
}

SYSTEMIC_SCENARIOS = [
    "Global Financial Crisis",
    "Market Liquidity Crisis",
    "Interest Rate +300bps",
    "Government Tightening",
    "Tariffs"
]

# 2) Preset PD theo MÃ (ưu tiên) & theo NGÀNH (mặc định)
#    PD dưới đây là ABSOLUTE PD dưới từng scenario (không cộng/trừ baseline)
PRESET_PD = {
    "ticker": {
        "HAG": {
            "baseline_min": 0.45,
            "sector": {"Credit Tightening": 0.78, "Property Price Correction": 0.72},
            "systemic": {
                "Global Financial Crisis": 0.85,
                "Market Liquidity Crisis": 0.82,
                "Interest Rate +300bps": 0.76,
                "Government Tightening": 0.70,
                "Tariffs": 0.62
            }
        },
        "ROS": {
            "baseline_min": 0.35,
            "sector": {"Credit Tightening": 0.60, "Property Price Correction": 0.55},
            "systemic": {
                "Global Financial Crisis": 0.70,
                "Market Liquidity Crisis": 0.66,
                "Interest Rate +300bps": 0.58,
                "Government Tightening": 0.52,
                "Tariffs": 0.40
            }
        },
        "HPG": {
            "baseline_min": 0.12,
            "sector": {"Steel Price Collapse": 0.40, "Energy Cost Surge": 0.32},
            "systemic": {
                "Global Financial Crisis": 0.45,
                "Market Liquidity Crisis": 0.38,
                "Interest Rate +300bps": 0.28,
                "Government Tightening": 0.24,
                "Tariffs": 0.22
            }
        },
        "VHM": {
            "baseline_min": 0.10,
            "sector": {"Credit Tightening": 0.38, "Property Price Correction": 0.34},
            "systemic": {
                "Global Financial Crisis": 0.42,
                "Market Liquidity Crisis": 0.40,
                "Interest Rate +300bps": 0.30,
                "Government Tightening": 0.28,
                "Tariffs": 0.18
            }
        },
        "VNM": {
            "baseline_min": 0.05,
            "sector": {"Energy Price Shock": 0.18, "Food Input Spike": 0.20},
            "systemic": {
                "Global Financial Crisis": 0.22,
                "Market Liquidity Crisis": 0.20,
                "Interest Rate +300bps": 0.14,
                "Government Tightening": 0.12,
                "Tariffs": 0.10
            }
        },
        "FPT": {
            "baseline_min": 0.06,
            "sector": {"Valuation Reset": 0.22, "Supply Chain Disruptions": 0.18},
            "systemic": {
                "Global Financial Crisis": 0.26,
                "Market Liquidity Crisis": 0.24,
                "Interest Rate +300bps": 0.16,
                "Government Tightening": 0.14,
                "Tariffs": 0.18
            }
        },
        "GAS": {
            "baseline_min": 0.08,
            "sector": {"Oil Demand Crash": 0.30, "Field Outage": 0.24},
            "systemic": {
                "Global Financial Crisis": 0.34,
                "Market Liquidity Crisis": 0.30,
                "Interest Rate +300bps": 0.20,
                "Government Tightening": 0.18,
                "Tariffs": 0.16
            }
        },
        "VJC": {
            "baseline_min": 0.15,
            "sector": {"Travel Collapse": 0.55, "Fuel Spike": 0.40},
            "systemic": {
                "Global Financial Crisis": 0.58,
                "Market Liquidity Crisis": 0.52,
                "Interest Rate +300bps": 0.34,
                "Government Tightening": 0.30,
                "Tariffs": 0.22
            }
        },
        "VIC": {
            "baseline_min": 0.12,
            "sector": {"COVID Demand Shock": 0.36, "Luxury Slowdown": 0.32},
            "systemic": {
                "Global Financial Crisis": 0.44,
                "Market Liquidity Crisis": 0.40,
                "Interest Rate +300bps": 0.30,
                "Government Tightening": 0.26,
                "Tariffs": 0.22
            }
        }
    },
    "sector": {
        "Real Estate": {"Credit Tightening": 0.42, "Property Price Correction": 0.36},
        "Materials": {"Steel Price Collapse": 0.34, "Energy Cost Surge": 0.28},
        "Energy": {"Oil Demand Crash": 0.30, "Field Outage": 0.24},
        "Technology": {"Valuation Reset": 0.24, "Supply Chain Disruptions": 0.20},
        "Consumer Discretionary": {"COVID Demand Shock": 0.32, "Luxury Slowdown": 0.26},
        "Consumer Staples": {"Energy Price Shock": 0.20, "Food Input Spike": 0.22},
        "Industrials": {"Logistics/Supply Chain": 0.26, "Export Order Drop": 0.22},
        "Utilities": {"Regulatory Tightening": 0.18},
        "Financials": {"Credit Loss Cycle": 0.28, "Funding Cost Rise": 0.24},
        "Healthcare": {"Reimbursement Pressure": 0.20},
        "Telecom": {"Capex Cycle Upswing": 0.20},
        "Transportation": {"Travel Collapse": 0.44, "Fuel Spike": 0.34},
        "Hospitality & Travel": {"Tourism Freeze": 0.48},
        "Agriculture & Fisheries": {"Export Shock": 0.24},
        "Automotive": {"Semiconductor Shortage": 0.30},
        "Other": {"Generic Sector Shock": 0.22}
    },
    "systemic": {
        "default": {
            "Global Financial Crisis": 0.38,
            "Market Liquidity Crisis": 0.34,
            "Interest Rate +300bps": 0.24,
            "Government Tightening": 0.22,
            "Tariffs": 0.18
        },
        "Real Estate": {
            "Global Financial Crisis": 0.45,
            "Market Liquidity Crisis": 0.40,
            "Interest Rate +300bps": 0.32,
            "Government Tightening": 0.28,
            "Tariffs": 0.20
        },
        "Materials": {
            "Global Financial Crisis": 0.42,
            "Market Liquidity Crisis": 0.36,
            "Interest Rate +300bps": 0.26,
            "Government Tightening": 0.24,
            "Tariffs": 0.24
        },
        "Transportation": {
            "Global Financial Crisis": 0.50,
            "Market Liquidity Crisis": 0.44,
            "Interest Rate +300bps": 0.34,
            "Government Tightening": 0.28,
            "Tariffs": 0.22
        }
    }
}

# 3) Lấy preset cho mã hiện tại
ticker_key = str(ticker).strip().upper()
bucket = sector_bucket if sector_bucket in SECTOR_SCENARIOS else "Other"

preset_ticker = PRESET_PD["ticker"].get(ticker_key, {})
preset_sector = PRESET_PD["sector"].get(bucket, PRESET_PD["sector"]["Other"])
preset_sys_bucket = PRESET_PD["systemic"].get(bucket, PRESET_PD["systemic"]["default"])

# Baseline: dùng PD sau điều chỉnh (B) nhưng không thấp hơn baseline_min của mã (nếu có)
baseline_min = float(preset_ticker.get("baseline_min", 0.0))
baseline_pd = float(max(pd_final, baseline_min))

# 4) Dựng bảng kết quả tuyệt đối theo scenario
sec_names = SECTOR_SCENARIOS.get(bucket, SECTOR_SCENARIOS["Other"])
abs_pd_sector = []
for nm in sec_names:
    val = preset_ticker.get("sector", {}).get(nm, preset_sector.get(nm, baseline_pd))
    abs_pd_sector.append((nm, float(val)))

abs_pd_systemic = []
for nm in SYSTEMIC_SCENARIOS:
    val = preset_ticker.get("systemic", {}).get(nm, preset_sys_bucket.get(nm, PRESET_PD["systemic"]["default"][nm]))
    abs_pd_systemic.append((nm, float(val)))

# 5) Impact % để vẽ
df_sector = pd.DataFrame(abs_pd_sector, columns=["Scenario", "PD"])
df_sector["Impact_%"] = (df_sector["PD"] - baseline_pd) / max(baseline_pd, 1e-9) * 100.0

df_sys = pd.DataFrame(abs_pd_systemic, columns=["Scenario", "PD"])
df_sys["Impact_%"] = (df_sys["PD"] - baseline_pd) / max(baseline_pd, 1e-9) * 100.0

st.caption(f"Sector raw: {sector_raw or '-'} → Bucket: **{bucket}** • Baseline PD (post-adj): **{baseline_pd:.2%}**")

# 6) Vẽ biểu đồ
c1, c2 = st.columns(2)
with c1:
    f1 = go.Figure()
    f1.add_trace(go.Bar(
        x=df_sector["Scenario"], y=df_sector["Impact_%"],
        text=[f"{v:+.1f}%" for v in df_sector["Impact_%"]],
        textposition="outside"
    ))
    f1.update_layout(
        title=f"Sector Impact — ΔPD vs Baseline (%) • {bucket}",
        yaxis=dict(title="Impact (%)"),
        height=340, margin=dict(l=10, r=10, t=48, b=80)
    )
    show_plotly(f1, f"sector_impact_preset_{ticker}_{year}")

with c2:
    f2 = go.Figure()
    f2.add_trace(go.Bar(
        x=df_sys["Scenario"], y=df_sys["Impact_%"],
        text=[f"{v:+.1f}%" for v in df_sys["Impact_%"]],
        textposition="outside"
    ))
    f2.update_layout(
        title="Systemic Impact — ΔPD vs Baseline (%)",
        yaxis=dict(title="Impact (%)"),
        height=340, margin=dict(l=10, r=10, t=48, b=80),
        xaxis_tickangle=-30
    )
    show_plotly(f2, f"systemic_impact_preset_{ticker}_{year}")

# 7) KPI tóm tắt (không Monte Carlo)
k1, k2 = st.columns(2)
with k1: st.metric("Baseline PD (post-adj)", f"{baseline_pd:.2%}")
with k2: st.metric(
    "Max PD under crises",
    f"{max(df_sector['PD'].max() if not df_sector.empty else 0.0, df_sys['PD'].max() if not df_sys.empty else 0.0):.2%}"
)

# 8) Bảng chi tiết
with st.expander("Scenario details"):
    out = pd.concat(
        [df_sector.assign(Type="Sector"), df_sys.assign(Type="Systemic")],
        ignore_index=True
    )[["Type", "Scenario", "PD", "Impact_%"]]
    out["PD"] = out["PD"].map(lambda v: f"{v:.2%}")
    out["Impact_%"] = out["Impact_%"].map(lambda v: f"{v:+.1f}%")
    st.dataframe(out, hide_index=True, use_container_width=True)

# ===================== E) Risk Commentary & Recommendation =====================
#
# Compose a qualitative assessment based on the PD, financial ratios and
# sentiment indicators. This section is only shown in the Summary tab.
st.subheader("E. Risk Commentary & Recommendation")

comment_lines = []
# 1) PD & policy band
comment_lines.append(
    f"Xác suất vỡ nợ (PD) theo mô hình là {pd_final:.2%}, tương ứng nhóm **{band}**."
)

# 2) Profitability
try:
    if isinstance(roa, float) and pd.notna(roa):
        if roa > 0:
            comment_lines.append("Hiệu suất sinh lời trên tài sản (ROA) dương, cho thấy doanh nghiệp tạo ra lợi nhuận từ tài sản của mình.")
        else:
            comment_lines.append("ROA âm, doanh nghiệp chưa tạo ra lợi nhuận từ tài sản.")
    if isinstance(roe, float) and pd.notna(roe):
        if roe > 0:
            comment_lines.append("ROE dương, vốn chủ sở hữu đang được sử dụng hiệu quả.")
        else:
            comment_lines.append("ROE âm, vốn chủ sở hữu chưa tạo ra lợi nhuận.")
except Exception:
    pass

# 3) Leverage
try:
    if isinstance(dte, float) and pd.notna(dte):
        if dte < 1:
            comment_lines.append("Đòn bẩy tài chính ở mức thấp (Debt/Equity < 1).")
        elif dte < 2:
            comment_lines.append("Đòn bẩy tài chính ở mức trung bình (1 ≤ Debt/Equity < 2).")
        else:
            comment_lines.append("Đòn bẩy tài chính cao (Debt/Equity ≥ 2), cần thận trọng.")
except Exception:
    pass

# 4) Liquidity
try:
    if isinstance(current_ratio, float) and pd.notna(current_ratio):
        if current_ratio > 1.5:
            comment_lines.append("Hệ số thanh khoản hiện hành cao, doanh nghiệp có khả năng thanh toán nợ ngắn hạn tốt.")
        elif current_ratio > 1.0:
            comment_lines.append("Hệ số thanh khoản hiện hành vừa phải, công ty cần quản lý dòng tiền cẩn thận.")
        else:
            comment_lines.append("Hệ số thanh khoản hiện hành thấp (<1), tiềm ẩn rủi ro thanh khoản.")
except Exception:
    pass

# 5) Sentiment
sentiment_val = None
try:
    for s_col in ["Sentiment Change", "sentiment change", "Sentiment_Change", "sentiment_change"]:
        if s_col in row_raw and pd.notna(row_raw.get(s_col)):
            sentiment_val = to_float(row_raw.get(s_col))
            break
    if sentiment_val is not None:
        if sentiment_val > 0.01:
            comment_lines.append("Tâm lý thị trường hiện đang **tích cực**, tin tức hỗ trợ giá cổ phiếu.")
        elif sentiment_val < -0.01:
            comment_lines.append("Tâm lý thị trường **tiêu cực**, có thể ảnh hưởng tới giá cổ phiếu.")
        else:
            comment_lines.append("Tâm lý thị trường **trung tính**, không có tin tức nổi bật.")
except Exception:
    pass

if comment_lines:
    for cl in comment_lines:
        st.write(f"- {cl}")
else:
    st.write("*Nội dung khuyến nghị sẽ được cập nhật tại đây.*")
