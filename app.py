import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import io
from datetime import datetime
import anthropic
from lmfit.models import LorentzianModel, VoigtModel, PseudoVoigtModel
from lmfit import Parameters
from scipy.signal import find_peaks

# Page configuration
st.set_page_config(
    page_title="Mössbauer Spectrum Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'fit_results' not in st.session_state:
    st.session_state.fit_results = None

class FitModel(Enum):
    LORENTZIAN = "lorentzian"
    VOIGT = "voigt"
    PSEUDO_VOIGT = "pseudo_voigt"

@dataclass
class MossbauerSite:
    isomer_shift: float
    quadrupole_splitting: float
    line_width: float
    relative_area: float
    site_type: str = ""
    hyperfine_field: Optional[float] = None

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

class MossbauerFitter:
    def __init__(self, model_type: FitModel):
        self.model_type = model_type
        self.velocity = None
        self.absorption = None
        self.result = None

    def load_data(self, uploaded_file) -> Tuple[bool, str]:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file).dropna(how='all')
                df.columns = df.columns.str.strip()  # Strip whitespace from column names
                if "Main" in df.columns and "Unnamed: 1" in df.columns:
                    df = df[["Main", "Unnamed: 1"]].dropna()
                    df.columns = ["velocity", "absorption"]
                else:
                    return False, "Excel file must contain columns labeled 'Main' and 'Unnamed: 1' (or similar)."
            elif uploaded_file.name.endswith(('.txt', '.csv')):
                content = uploaded_file.read().decode('utf-8')
                uploaded_file.seek(0)
                for sep in ['\t', ',', ' ', ';']:
                    try:
                        df = pd.read_csv(io.StringIO(content), sep=sep, header=None)
                        if df.shape[1] >= 2:
                            break
                    except:
                        continue
                df = df.dropna().reset_index(drop=True)
                df.columns = ["velocity", "absorption"] + list(df.columns[2:])
            else:
                return False, "Unsupported file format. Please use .xlsx, .txt, or .csv"

            if df.shape[0] < 10:
                return False, "Insufficient data points (minimum 10 required)"

            self.velocity = df["velocity"].astype(float).values
            self.absorption = df["absorption"].astype(float).values

            if self.absorption.max() > 10:
                self.absorption = self.absorption / 100.0

            if self.absorption.min() < 0.9:
                baseline = np.percentile(self.absorption, 95)
                self.absorption = self.absorption / baseline

            return True, "Data loaded successfully"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"

def main():
    st.sidebar.subheader("🔑 Anthropic API Key")
    api_key_input = st.sidebar.text_input("Enter Claude API key", type="password", value=st.session_state.api_key or "")
    if api_key_input and api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.sidebar.success("API key set!")


    st.title("⚛️ Mössbauer Spectrum Analyzer")

    st.sidebar.subheader("⚙️ Fitting Options")
    selected_model = st.sidebar.selectbox(
        "Fitting model:",
        options=list(FitModel),
        format_func=lambda x: x.value.title(),
        index=0
    )

    n_sites = st.sidebar.number_input(
        "Number of Mössbauer sites:",
        min_value=1,
        max_value=6,
        value=2,
        help="Estimated number of iron environments to fit"
    )
    st.markdown("AI-powered analysis of ⁵⁷Fe Mössbauer spectroscopy data")

    uploaded_file = st.file_uploader("Upload Mössbauer spectrum file (.xlsx, .csv, .txt)", type=["xlsx", "csv", "txt"])

    if uploaded_file is not None:
        with st.spinner("Loading data..."):
            fitter = MossbauerFitter(model_type=FitModel.LORENTZIAN)
            success, message = fitter.load_data(uploaded_file)
            if not success:
                st.error(message)
                return
            st.success(message)

            st.line_chart(pd.DataFrame({"absorption": fitter.absorption}, index=fitter.velocity))
            st.info("✅ Data loaded and plotted. Continue integration with fitting logic here.")
    else:
        st.info("📁 Please upload a file to begin analysis.")

if __name__ == "__main__":
    main()
