"""
Financial Analysis Tab
=======================

This module implements the Financial Analysis tab for the default risk
dashboard.  It builds income statement, balance sheet, cash flow and
ratio tables directly from the raw financial data provided in
`bctc_final.csv`.  All values are aggregated per‐year for the selected
ticker so that users can compare trends over time.  At the bottom of
the view a short narrative note is generated based on observed trends
in revenue, net profit, leverage and liquidity.  Labels and section
headings are translated on the fly based on the `lang` argument
(`'vi'` for Vietnamese, `'en'` for English).

The `render` function is the only public entry point.  It accepts the
feature dataframe, raw dataframe, ticker, year, sector and language
code as inputs and draws tables and charts using Streamlit.  The
implementation avoids any hard–coded sample numbers – instead it
computes values from the underlying data, performing safe numeric
conversions where necessary.  If data for a particular item is not
available the corresponding cell is left blank.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _extract_value(row: pd.Series, cols: list[str]) -> float | None:
    """Return the first non-null numeric value from a list of possible columns.

    Many financial metrics may be stored under different column names across
    datasets.  This helper tries each name in order and converts the value to
    float, returning NaN if none are found.
    """
    for col in cols:
        if col in row and pd.notna(row[col]):
            try:
                val = row[col]
                if isinstance(val, str):
                    # remove commas and spaces for string numbers
                    val = val.replace(",", "")
                return float(val)
            except Exception:
                pass
    return np.nan


def _safe_div(a: float | None, b: float | None) -> float | np.nan:
    """Safely divide two values, returning NaN on zero or missing denominators."""
    try:
        if b is None or (isinstance(b, float) and not np.isfinite(b)) or b == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def _fmt_money(x: float | None) -> str:
    """Format a numeric value as currency with comma separators."""
    return "-" if (x is None or not np.isfinite(x)) else f"{x:,.1f}"


def _fmt_ratio(x: float | None) -> str:
    """Format a numeric ratio as percentage where appropriate."""
    if x is None or not np.isfinite(x):
        return "-"
    # treat values between -1.5 and 1.5 as percentages
    return f"{x:.2%}" if -1.5 <= float(x) <= 1.5 else f"{x:,.3f}"


# -----------------------------------------------------------------------------
# Translation dictionaries
# -----------------------------------------------------------------------------

# Vietnamese and English translations for table headers and metrics.  Each
# dictionary maps the internal English key to the label shown in the UI for
# that language.  Feel free to extend these dictionaries as needed; any keys
# missing will fall back to the English label.
VI_LABELS = {
    # Section titles
    "income_statement": "Báo cáo kết quả kinh doanh",
    "balance_sheet": "Bảng cân đối kế toán",
    "cash_flow": "Báo cáo lưu chuyển tiền tệ",
    "ratios": "Tỷ số tài chính",
    "notes": "Ghi chú & Phân tích",
    # Column headers
    "Revenue": "Doanh thu thuần",
    "Gross Profit": "Lợi nhuận gộp",
    "Operating Profit": "Lợi nhuận từ hoạt động",
    "Profit Before Tax": "Lợi nhuận trước thuế",
    "Net Profit": "Lợi nhuận ròng",
    "Cash & CE": "Tiền mặt & TDT",
    "Accounts Receivable": "Khoản phải thu",
    "Inventories": "Hàng tồn kho",
    "Other Current Assets": "Tài sản lưu động khác",
    "Long-term Assets": "Tài sản dài hạn",
    "Fixed Assets": "Tài sản cố định",
    "Long-term Investments": "Đầu tư dài hạn",
    "Other Non-current Assets": "Tài sản dài hạn khác",
    "Total Assets": "Tổng tài sản",
    "Total Liabilities": "Tổng nợ phải trả",
    "Current Liabilities": "Nợ ngắn hạn",
    "Long-term Liabilities": "Nợ dài hạn",
    "Equity": "Vốn chủ sở hữu",
    "Operating CF": "Dòng tiền HĐKD",
    "Investing CF": "Dòng tiền đầu tư",
    "Financing CF": "Dòng tiền tài chính",
    "Net Change in Cash": "Thay đổi tiền mặt ròng",
    "Ending Cash": "Tiền cuối kỳ",
    "ROA": "ROA",
    "ROE": "ROE",
    "Debt_to_Assets": "Nợ/Tài sản",
    "Debt_to_Equity": "Nợ/Vốn chủ",
    "Current_Ratio": "Hệ số thanh khoản hiện tại",
    "Quick_Ratio": "Hệ số thanh khoản nhanh",
    # Note section
    "note_revenue_up": "Doanh thu tăng trưởng ổn định trong giai đoạn gần đây.",
    "note_revenue_down": "Doanh thu có xu hướng giảm đáng kể so với những năm trước.",
    "note_revenue_stable": "Doanh thu dao động nhẹ nhưng không thay đổi nhiều.",
    "note_profit_up": "Lợi nhuận ròng cải thiện qua thời gian, cho thấy hoạt động hiệu quả hơn.",
    "note_profit_down": "Lợi nhuận ròng suy giảm, cần xem xét nguyên nhân chi phí hoặc doanh thu.",
    "note_profit_stable": "Lợi nhuận ròng ổn định qua các năm.",
    "note_leverage_low": "Tỷ lệ nợ trên vốn chủ sở hữu ở mức an toàn (<1), cho thấy khả năng tự tài trợ tốt.",
    "note_leverage_medium": "Tỷ lệ nợ trên vốn chủ sở hữu trung bình, công ty dùng đòn bẩy nợ tương đối.",
    "note_leverage_high": "Tỷ lệ nợ trên vốn chủ sở hữu cao, tiềm ẩn rủi ro đòn bẩy tài chính.",
    "note_liquidity_high": "Hệ số thanh khoản hiện hành cao (>1.5), khả năng đáp ứng nợ ngắn hạn tốt.",
    "note_liquidity_medium": "Hệ số thanh khoản hiện hành ở mức trung bình, cần theo dõi.",
    "note_liquidity_low": "Hệ số thanh khoản hiện hành thấp (<1), công ty có thể gặp khó khăn trong thanh toán ngắn hạn.",
}

EN_LABELS = {
    "income_statement": "Income Statement",
    "balance_sheet": "Balance Sheet",
    "cash_flow": "Cash Flow Statement",
    "ratios": "Financial Ratios",
    "notes": "Notes & Analysis",
    # Column headers
    "Revenue": "Revenue",
    "Gross Profit": "Gross Profit",
    "Operating Profit": "Operating Profit",
    "Profit Before Tax": "Profit Before Tax",
    "Net Profit": "Net Profit",
    "Cash & CE": "Cash & CE",
    "Accounts Receivable": "Accounts Receivable",
    "Inventories": "Inventories",
    "Other Current Assets": "Other Current Assets",
    "Long-term Assets": "Long-term Assets",
    "Fixed Assets": "Fixed Assets",
    "Long-term Investments": "Long-term Investments",
    "Other Non-current Assets": "Other Non-current Assets",
    "Total Assets": "Total Assets",
    "Total Liabilities": "Total Liabilities",
    "Current Liabilities": "Current Liabilities",
    "Long-term Liabilities": "Long-term Liabilities",
    "Equity": "Equity",
    "Operating CF": "Operating Cash Flow",
    "Investing CF": "Investing Cash Flow",
    "Financing CF": "Financing Cash Flow",
    "Net Change in Cash": "Net Change in Cash",
    "Ending Cash": "Ending Cash",
    "ROA": "ROA",
    "ROE": "ROE",
    "Debt_to_Assets": "Debt to Assets",
    "Debt_to_Equity": "Debt to Equity",
    "Current_Ratio": "Current Ratio",
    "Quick_Ratio": "Quick Ratio",
    # Notes
    "note_revenue_up": "Revenue has been growing steadily over recent years.",
    "note_revenue_down": "Revenue shows a significant downward trend compared to prior years.",
    "note_revenue_stable": "Revenue fluctuates only slightly with no major change.",
    "note_profit_up": "Net profit has improved over time, indicating better operating efficiency.",
    "note_profit_down": "Net profit is declining; investigate cost or revenue drivers.",
    "note_profit_stable": "Net profit remains stable across the years.",
    "note_leverage_low": "Debt to equity ratio is low (<1), indicating strong self–funding capability.",
    "note_leverage_medium": "Debt to equity ratio is moderate; the company uses some leverage.",
    "note_leverage_high": "Debt to equity ratio is high, signalling leverage risk.",
    "note_liquidity_high": "Current ratio is high (>1.5), indicating good short-term solvency.",
    "note_liquidity_medium": "Current ratio is average; liquidity should be monitored.",
    "note_liquidity_low": "Current ratio is low (<1), suggesting potential short-term liquidity issues.",
}


def _t(label_key: str, lang: str) -> str:
    """Return the translated label for the given key and language."""
    if lang == 'vi':
        return VI_LABELS.get(label_key, label_key)
    return EN_LABELS.get(label_key, label_key)


# -----------------------------------------------------------------------------
# Main render function
# -----------------------------------------------------------------------------

def render(feats_df: pd.DataFrame, raw_df: pd.DataFrame, ticker: str, year: int,
           sector: str, lang: str = 'vi') -> None:
    """
    Render the Finance tab.

    Parameters
    ----------
    feats_df : pd.DataFrame
        Feature dataframe (unused but kept for API consistency).
    raw_df : pd.DataFrame
        Raw financial data loaded from `bctc_final.csv`.
    ticker : str
        The ticker symbol selected in the sidebar.
    year : int
        The latest year selected; used for context but all years are plotted.
    sector : str
        Sector classification of the company (for display only).
    lang : str, optional
        Language code ('vi' for Vietnamese, 'en' for English).
    """
    st.subheader(f"📊 { _t('income_statement', lang) } / { _t('balance_sheet', lang) }")

    # Filter historical records for the selected ticker and sort by year
    hist = raw_df[raw_df["Ticker"].astype(str) == ticker].copy()
    if hist.empty:
        st.info(
            "Không có dữ liệu" if lang == 'vi' else "No financial data available for this ticker."
        )
        return
    hist = hist.sort_values("Year").reset_index(drop=True)
    hist["Year"] = pd.to_numeric(hist["Year"], errors="coerce")

    # Construct Income Statement across years
    income_items = [
        ("Revenue", ["Net Sales", "Revenue (Bn. VND)", "Revenue"]),
        ("Gross Profit", ["Gross Profit"]),
        ("Operating Profit", ["Operating Profit/Loss", "Operating_Profit"]),
        ("Profit Before Tax", ["Profit before tax", "Net Profit/Loss before tax"]),
        ("Net Profit", ["Net Profit For the Year", "Net_Profit"]),
    ]
    income_data: list[dict[str, float | int | None]] = []
    for _, row in hist.iterrows():
        yr = int(row.get("Year")) if pd.notna(row.get("Year")) else None
        row_dict: dict[str, float | int | None] = {"Year": yr}
        for name, cols in income_items:
            row_dict[name] = _extract_value(row, cols)
        income_data.append(row_dict)
    income_df = pd.DataFrame(income_data)
    if not income_df.empty:
        income_df = income_df.set_index("Year").sort_index()

    # Construct Balance Sheet across years
    bs_items = [
        ("Cash & CE", ["Cash and cash equivalents (Bn. VND)", "Cash and cash equivalents", "Cash"]),
        ("Accounts Receivable", ["Accounts receivable (Bn. VND)", "Short-term loans receivables (Bn. VND)"]),
        ("Inventories", ["Net Inventories", "Inventories", "Net Inventories, and other inventory categories"]),
        ("Other Current Assets", ["Other current assets", "Other current assets (Bn. VND)"]),
        ("Long-term Assets", ["LONG-TERM ASSETS (Bn. VND)", "Long_term_assets"]),
        ("Fixed Assets", ["Fixed assets (Bn. VND)", "Fixed assets"]),
        ("Long-term Investments", ["Long-term investments (Bn. VND)"]),
        ("Other Non-current Assets", ["Other non-current assets", "Other non-current assets (Bn. VND)"]),
        ("Total Assets", ["TOTAL ASSETS (Bn. VND)", "Total_Assets"]),
        ("Total Liabilities", ["LIABILITIES (Bn. VND)", "Total liabilities (Bn. VND)"]),
        ("Current Liabilities", ["Current liabilities (Bn. VND)", "Current_Liabilities"]),
        ("Long-term Liabilities", ["Long-term liabilities (Bn. VND)", "Long_Term_Liabilities"]),
        ("Equity", ["OWNER'S EQUITY(Bn.VND)", "Equity"]),
    ]
    bs_data: list[dict[str, float | int | None]] = []
    for _, row in hist.iterrows():
        yr = int(row.get("Year")) if pd.notna(row.get("Year")) else None
        row_dict: dict[str, float | int | None] = {"Year": yr}
        for name, cols in bs_items:
            row_dict[name] = _extract_value(row, cols)
        bs_data.append(row_dict)
    bs_df = pd.DataFrame(bs_data)
    if not bs_df.empty:
        bs_df = bs_df.set_index("Year").sort_index()

    # Construct Cash Flow across years
    cf_items = [
        ("Operating CF", ["Net cash inflows/outflows from operating activities", "Net cash flows from operating activities"]),
        ("Investing CF", ["Net Cash Flows from Investing Activities", "Net cash flows from investing activities"]),
        ("Financing CF", ["Cash flows from financial activities", "Net cash flows from financing activities"]),
        ("Net Change in Cash", ["Net increase/decrease in cash and cash equivalents", "Net cash flows (increase/decrease) in cash"]),
        ("Ending Cash", ["Cash and Cash Equivalents at the end of period", "Cash and cash equivalents"]),
    ]
    cf_data: list[dict[str, float | int | None]] = []
    for _, row in hist.iterrows():
        yr = int(row.get("Year")) if pd.notna(row.get("Year")) else None
        row_dict: dict[str, float | int | None] = {"Year": yr}
        for name, cols in cf_items:
            row_dict[name] = _extract_value(row, cols)
        cf_data.append(row_dict)
    cf_df = pd.DataFrame(cf_data)
    if not cf_df.empty:
        cf_df = cf_df.set_index("Year").sort_index()

    # Construct ratios across years
    ratio_rows: list[dict[str, float | int | None]] = []
    for _, row in hist.iterrows():
        yr = int(row.get("Year")) if pd.notna(row.get("Year")) else None
        assets = _extract_value(row, ["TOTAL ASSETS (Bn. VND)", "Total_Assets"])
        equity = _extract_value(row, ["OWNER'S EQUITY(Bn.VND)", "Equity"])
        curr_liab = _extract_value(row, ["Current liabilities (Bn. VND)", "Current_Liabilities"])
        long_liab = _extract_value(row, ["Long-term liabilities (Bn. VND)", "Long_Term_Liabilities"])
        debt = _extract_value(row, ["Total_Debt", "Long-term borrowings (Bn. VND)", "Short-term borrowings (Bn. VND)"])
        revenue = _extract_value(row, ["Net Sales", "Revenue (Bn. VND)", "Revenue"])
        net_profit = _extract_value(row, ["Net Profit For the Year", "Net_Profit"])
        cash_val = _extract_value(row, ["Cash and cash equivalents (Bn. VND)", "Cash and cash equivalents", "Cash"])
        receivables_val = _extract_value(row, ["Accounts receivable (Bn. VND)", "Short-term loans receivables (Bn. VND)"])
        current_assets = _extract_value(row, ["CURRENT ASSETS (Bn. VND)", "Current_Assets"])
        total_liab = (curr_liab or 0.0) + (long_liab or 0.0)
        r_roa = _safe_div(net_profit, assets)
        r_roe = _safe_div(net_profit, equity)
        r_dta = _safe_div(total_liab, assets)
        r_dte = _safe_div(debt, equity)
        r_cr = _safe_div(current_assets, curr_liab)
        r_qr = _safe_div((cash_val or 0.0) + (receivables_val or 0.0), curr_liab)
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

    # -------------------------------------------------------------------------
    # Display sections
    # -------------------------------------------------------------------------

    # Income Statement
    st.subheader(_t("income_statement", lang))
    if income_df.empty:
        st.info(
            "Không có dữ liệu báo cáo kết quả kinh doanh" if lang == 'vi' else "No income statement data found."
        )
    else:
        # rename columns based on language and format numbers
        disp = income_df.copy()
        disp = disp.applymap(lambda x: _fmt_money(x) if pd.notna(x) else "-")
        disp.index = disp.index.astype(int)
        disp.rename(columns={col: _t(col, lang) for col in disp.columns}, inplace=True)
        st.dataframe(disp, use_container_width=True)

    # Balance Sheet
    st.subheader(_t("balance_sheet", lang))
    if bs_df.empty:
        st.info(
            "Không có dữ liệu bảng cân đối" if lang == 'vi' else "No balance sheet data found."
        )
    else:
        disp = bs_df.copy()
        disp = disp.applymap(lambda x: _fmt_money(x) if pd.notna(x) else "-")
        disp.index = disp.index.astype(int)
        disp.rename(columns={col: _t(col, lang) for col in disp.columns}, inplace=True)
        st.dataframe(disp, use_container_width=True)

    # Cash Flow
    st.subheader(_t("cash_flow", lang))
    if cf_df.empty:
        st.info(
            "Không có dữ liệu lưu chuyển tiền tệ" if lang == 'vi' else "No cash flow statement data found."
        )
    else:
        disp = cf_df.copy()
        disp = disp.applymap(lambda x: _fmt_money(x) if pd.notna(x) else "-")
        disp.index = disp.index.astype(int)
        disp.rename(columns={col: _t(col, lang) for col in disp.columns}, inplace=True)
        st.dataframe(disp, use_container_width=True)

    # Ratios
    st.subheader(_t("ratios", lang))
    if ratio_df.empty:
        st.info(
            "Không có dữ liệu tỷ số tài chính" if lang == 'vi' else "No ratio data found."
        )
    else:
        disp = ratio_df.copy()
        disp.index = disp.index.astype(int)
        # pivot so metrics are rows and years are columns for readability
        pivot = disp.T
        pivot = pivot.applymap(lambda x: _fmt_ratio(x) if pd.notna(x) else "-")
        pivot.rename(index={idx: _t(idx, lang) for idx in pivot.index}, inplace=True)
        st.dataframe(pivot, use_container_width=True)

    # Revenue & Profit Trend Chart
    if not income_df.empty:
        try:
            years = income_df.index.astype(int).tolist()
            revenues = income_df["Revenue"].tolist()
            profits = income_df["Net Profit"].tolist()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=years, y=revenues, name=_t("Revenue", lang)))
            fig.add_trace(go.Scatter(x=years, y=profits, name=_t("Net Profit", lang), mode="lines+markers", yaxis="y2"))
            fig.update_layout(
                title=_t("Revenue", lang) + " & " + _t("Net Profit", lang),
                yaxis=dict(title=_t("Revenue", lang)),
                yaxis2=dict(title=_t("Net Profit", lang), overlaying="y", side="right"),
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Generate narrative notes
    # -------------------------------------------------------------------------
    notes: list[str] = []
    # Revenue trend
    if not income_df.empty and not income_df["Revenue"].dropna().empty:
        rev_series = income_df["Revenue"].dropna()
        if not rev_series.empty:
            rev_start = rev_series.iloc[0]; rev_end = rev_series.iloc[-1]
            if np.isfinite(rev_start) and np.isfinite(rev_end):
                if rev_end > rev_start * 1.05:
                    notes.append(_t("note_revenue_up", lang))
                elif rev_end < rev_start * 0.95:
                    notes.append(_t("note_revenue_down", lang))
                else:
                    notes.append(_t("note_revenue_stable", lang))
    # Profit trend
    if not income_df.empty and not income_df["Net Profit"].dropna().empty:
        pr_series = income_df["Net Profit"].dropna()
        if not pr_series.empty:
            pr_start = pr_series.iloc[0]; pr_end = pr_series.iloc[-1]
            if np.isfinite(pr_start) and np.isfinite(pr_end):
                if pr_end > pr_start * 1.05:
                    notes.append(_t("note_profit_up", lang))
                elif pr_end < pr_start * 0.95:
                    notes.append(_t("note_profit_down", lang))
                else:
                    notes.append(_t("note_profit_stable", lang))
    # Leverage level
    if not ratio_df.empty and not ratio_df["Debt_to_Equity"].dropna().empty:
        dte_latest = ratio_df["Debt_to_Equity"].dropna().iloc[-1]
        if np.isfinite(dte_latest):
            if dte_latest < 1:
                notes.append(_t("note_leverage_low", lang))
            elif dte_latest < 2:
                notes.append(_t("note_leverage_medium", lang))
            else:
                notes.append(_t("note_leverage_high", lang))
    # Liquidity level
    if not ratio_df.empty and not ratio_df["Current_Ratio"].dropna().empty:
        cr_latest = ratio_df["Current_Ratio"].dropna().iloc[-1]
        if np.isfinite(cr_latest):
            if cr_latest > 1.5:
                notes.append(_t("note_liquidity_high", lang))
            elif cr_latest > 1.0:
                notes.append(_t("note_liquidity_medium", lang))
            else:
                notes.append(_t("note_liquidity_low", lang))
    # Display notes
    st.subheader(_t("notes", lang))
    if not notes:
        st.write(
            "*Chưa có ghi chú; bạn có thể bổ sung tại đây.*" if lang == 'vi' else "*No notes generated; you may add your own commentary here.*"
        )
    else:
        for line in notes:
            st.write(f"- {line}")
