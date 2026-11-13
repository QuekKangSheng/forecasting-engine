"""App entrypoint: the frame every page sits in, and the sidebar navigation.

Run with: uv run streamlit run app/Home.py
"""

import streamlit as st

import shell
import ui
import warmup
from forecasting_engine.store import _db

# Every page reads the store on every rerun, and opening the file was most of what
# each read cost: hold it open for as long as the app runs.
_db.keep_open()
# The libraries the other pages need, loaded behind this one while it is read.
warmup.start()

st.set_page_config(
    page_title="Forecasting Engine", page_icon=":material/monitoring:", layout="wide"
)

# In pipeline order, which the sidebar numbers: data in, forecasts out of it, an
# allocation out of those, then what would move it and what has been run.
pages = {
    "home": st.Page("app_pages/0_Home.py", title="Home", default=True),
    "data": st.Page("app_pages/1_Data.py", title="Data"),
    "models": st.Page("app_pages/2_Models.py", title="Models"),
    "portfolio": st.Page("app_pages/3_Portfolio.py", title="Portfolio Optimizer"),
    "scenarios": st.Page("app_pages/4_Scenarios.py", title="Scenarios"),
    "history": st.Page("app_pages/5_Run_History.py", title="Run history"),
}
# So a page can link to the next one without knowing where it lives.
ui.register_pages(pages)
pg = st.navigation(
    {
        "": [pages["home"]],
        "Pipeline": [pages["data"], pages["models"], pages["portfolio"]],
        "Analysis": [pages["scenarios"], pages["history"]],
    }
)
shell.frame(pg.run)
