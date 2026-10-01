"""Home page: overview and the data quality report.

The report is here rather than behind a settings menu because it is what you
check before trusting a forecast built on this data.
"""

import streamlit as st

import bloomberg_extraction_panel

st.title("Forecasting Engine")
st.write(
    "Forecasts short-horizon returns for liquid equity and bond indices, "
    "validates them against overfitting, and reports tail risk."
)

st.divider()
bloomberg_extraction_panel.render_summary()
