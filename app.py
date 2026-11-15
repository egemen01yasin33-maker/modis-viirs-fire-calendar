"""ASGI entry for Cursor publish host (and uvicorn).

Local Streamlit:  streamlit run streamlit_app.py
ASGI / publish:   uvicorn app:app --host 0.0.0.0 --port ${PORT:-8529}
"""

from __future__ import annotations

import streamlit as st

# Cursor publish expects a top-level app / application / handler.
app = st.App("streamlit_app.py")
application = app
handler = app
