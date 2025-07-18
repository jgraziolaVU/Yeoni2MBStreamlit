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
                df.columns = df.columns.str.strip()
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

class MossbauerInterpreter:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def generate_summary(self, velocity: np.ndarray, absorption: np.ndarray) -> str:
        client = anthropic.Anthropic(api_key=self.api_key)

        values = [f"{v:.3f} {a:.3f}" for v, a in zip(velocity, absorption)]
        prompt = """You are an expert in Mössbauer spectroscopy. Here is a velocity vs. absorption spectrum from a ⁵⁷Fe sample. Provide a brief interpretation of the number and nature of iron sites based on the shape and number of peaks.

Data (velocity [mm/s] absorption):
""" + "\n".join(values[:60]) + ("\n..." if len(values) > 60 else "")

        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )

        return message.content[0].text.strip()

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
            fitter = MossbauerFitter(model_type=selected_model)
            success, message = fitter.load_data(uploaded_file)
            if not success:
                st.error(message)
                return
            st.success(message)

            st.line_chart(pd.DataFrame({"absorption": fitter.absorption}, index=fitter.velocity))
            st.info("✅ Data loaded and plotted.")

            if st.session_state.api_key:
                with st.spinner("Generating Claude interpretation..."):
                    interpreter = MossbauerInterpreter(api_key=st.session_state.api_key)
                    summary = interpreter.generate_summary(fitter.velocity, fitter.absorption)
                    st.subheader("🧠 Claude Interpretation")
                    st.info(summary)
            else:
                st.warning("Set your Anthropic API key in the sidebar to enable AI interpretation.")
    else:
        st.info("📁 Please upload a file to begin analysis.")

if __name__ == "__main__":
    main()
