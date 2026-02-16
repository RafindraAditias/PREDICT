import streamlit as st

def inject_css():
    st.markdown("""
    <style>
      .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }
      [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid rgba(15, 23, 42, 0.06); }
      button[role="tab"] { font-weight: 700; }
      .card {
        background: #FFFFFF;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 6px 18px rgba(2, 6, 23, 0.06);
      }
      .muted { color: rgba(15, 23, 42, 0.62); font-size: 0.92rem; }
      .title { font-size: 1.05rem; font-weight: 800; color: #0F172A; margin: 0; }
      .big { font-size: 2.0rem; font-weight: 900; color: #0F172A; line-height: 1.1; margin: 0; }
      .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.82rem;
        border: 1px solid rgba(15, 23, 42, 0.10);
      }
      .pill.baik { background: rgba(34,197,94,0.14); color: #166534; }
      .pill.sedang { background: rgba(245,158,11,0.18); color: #92400E; }
      .pill.tidak-sehat { background: rgba(251,191,36,0.18); color: #78350F; }
      .pill.sangat-tidak-sehat { background: rgba(239,68,68,0.14); color: #991B1B; }
      .pill.berbahaya { background: rgba(127,29,29,0.14); color: #7F1D1D; }
      [data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; border: 1px solid rgba(15, 23, 42, 0.08); }
      div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 16px;
        padding: 10px 12px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.06);
      }
    </style>
    """, unsafe_allow_html=True)

def pill_class(category: str):
    """
    Map kategori ke class CSS untuk styling pill.
    """
    c = (category or "").lower()
    
    if "baik" in c:
        return "baik"
    if "sedang" in c:
        return "sedang"
    if "tidak sehat" in c and "sangat" not in c:
        return "tidak-sehat"
    if "sangat tidak sehat" in c:
        return "sangat-tidak-sehat"
    if "berbahaya" in c:
        return "berbahaya"
    
    # Fallback untuk kategori bahasa Inggris (kalau masih ada)
    if "good" in c:
        return "baik"
    if "moderate" in c:
        return "sedang"
    if "unhealthy" in c:
        return "tidak-sehat"
    if "hazardous" in c:
        return "berbahaya"
    
    return "baik"