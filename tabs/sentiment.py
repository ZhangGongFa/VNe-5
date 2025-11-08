"""
Sentiment Analysis Tab
======================

This module implements the Sentiment Analysis tab.  The dataset
`bctc_final.csv` may contain basic sentiment metrics such as "Sentiment
Change" or "News Shock" columns.  The `render` function extracts these
metrics for the selected ticker across all years, classifies each value
into Positive/Negative/Neutral categories, and presents a summary table
and simple line chart.  A placeholder section is provided for users to
add recent news headlines and commentary.  All text labels are
translated according to the supplied language code (`lang`).  If
sentiment metrics are not present in the dataset, a helpful message
informs the user.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime


# -----------------------------------------------------------------------------
# Translation dictionaries for the Sentiment tab
# -----------------------------------------------------------------------------

VI_LABELS = {
    "recent_news": "Tin tức gần đây",
    "sentiment_metrics": "Chỉ số sentiment theo năm",
    "no_sentiment_data": "Không có dữ liệu sentiment cho mã cổ phiếu này.",
    "metric": "Chỉ số",
    "classification": "Phân loại",
    "positive": "Tích cực",
    "negative": "Tiêu cực",
    "neutral": "Trung lập",
    "news_and_tone": "Tin tức & Nhận định thị trường",
    "placeholder": "Hiện tại chưa có dữ liệu tin tức chi tiết. Bạn có thể thêm danh sách tiêu đề tin tức và nhận định của mình tại đây.",
}

EN_LABELS = {
    "recent_news": "Recent News",
    "sentiment_metrics": "Sentiment metrics by year",
    "no_sentiment_data": "No sentiment data available for this ticker.",
    "metric": "Metric",
    "classification": "Classification",
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "news_and_tone": "News & Market Tone",
    "placeholder": "No detailed news data is available. You may add your own headlines and commentary here.",
}


def _t(key: str, lang: str) -> str:
    """Translate a key based on language."""
    return VI_LABELS.get(key, key) if lang == 'vi' else EN_LABELS.get(key, key)


def render(feats_df: pd.DataFrame, raw_df: pd.DataFrame, ticker: str, year: int,
           sector: str, lang: str = 'vi') -> None:
    """
    Render the Sentiment Analysis tab.

    Parameters
    ----------
    feats_df : pd.DataFrame
        Feature dataframe (unused but retained for API consistency).
    raw_df : pd.DataFrame
        Raw financial dataset; may contain sentiment columns.
    ticker : str
        The selected ticker.
    year : int
        The selected year (unused for metrics, but part of context).
    sector : str
        Sector name for display.
    lang : str
        Language code ('vi' or 'en').
    """
    st.subheader("📰 " + (_t("recent_news", lang) if lang == 'vi' else _t("recent_news", lang)))

    # Extract sentiment columns (case-insensitive match for 'sentiment' or 'news shock')
    hist = raw_df[raw_df["Ticker"].astype(str) == ticker].copy()
    if hist.empty:
        st.info(
            "Không có dữ liệu sentiment" if lang == 'vi' else "No sentiment data available."
        )
        return
    hist = hist.sort_values("Year")
    hist["Year"] = pd.to_numeric(hist["Year"], errors="coerce")
    sentiment_cols = [c for c in hist.columns if c.lower().strip().replace(" ", "") in ["sentimentchange", "newsshock"]]

    # Display sentiment metrics if available
    st.subheader(_t("sentiment_metrics", lang))
    if not sentiment_cols:
        st.info(_t("no_sentiment_data", lang))
    else:
        df_sent = hist[["Year"] + sentiment_cols].dropna()
        if df_sent.empty:
            st.info(_t("no_sentiment_data", lang))
        else:
            df_sent = df_sent.set_index("Year").sort_index()
            # classification for each metric
            def classify(val: float) -> str:
                try:
                    v = float(val)
                    if v > 0.01:
                        return _t("positive", lang)
                    if v < -0.01:
                        return _t("negative", lang)
                    return _t("neutral", lang)
                except Exception:
                    return _t("neutral", lang)

            display_df = df_sent.copy()
            # Add classification columns for each sentiment metric
            for col in sentiment_cols:
                display_df[f"{col}_class"] = display_df[col].apply(classify)
            # Format the numeric values to 4 decimal places
            for col in sentiment_cols:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
            # Rename columns for display
            rename_dict = {}
            for col in sentiment_cols:
                # Human friendly column names with spaces
                pretty = col.replace("_", " ").title()
                rename_dict[col] = pretty
                rename_dict[f"{col}_class"] = f"{pretty} ({_t('classification', lang)})"
            display_df.rename(columns=rename_dict, inplace=True)
            st.dataframe(display_df, use_container_width=True)
            # plot the first sentiment metric over time
            try:
                first_col = sentiment_cols[0]
                y_vals = hist.set_index("Year")[first_col].dropna()
                if not y_vals.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=y_vals.index.astype(int),
                        y=y_vals.values,
                        mode="lines+markers",
                        name=first_col,
                        line=dict(color="#1f77b4"),
                    ))
                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig.update_layout(
                        title=f"{first_col}"
                            if lang == 'en' else f"{first_col}" ,
                        xaxis_title="Year", yaxis_title=first_col,
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass

    # News & Market Tone placeholder
    st.subheader(_t("news_and_tone", lang))
    st.write(_t("placeholder", lang))
