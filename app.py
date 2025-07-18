import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import io
from datetime import datetime
import anthropic
from lmfit.models import LorentzianModel, VoigtModel, PseudoVoigtModel
from lmfit import Parameters
from scipy.signal import find_peaks, savgol_filter
import warnings
warnings.filterwarnings('ignore')

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
if 'claude_model' not in st.session_state:
    st.session_state.claude_model = "claude-sonnet-4-20250514"

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

@dataclass
class FitStatistics:
    r_squared: float
    reduced_chi_square: float
    aic: float
    bic: float
    rmse: float
    n_parameters: int
    n_data_points: int

class DataValidator:
    @staticmethod
    def validate_file_format(uploaded_file) -> Tuple[bool, str]:
        """Validate file format and basic structure"""
        try:
            filename = uploaded_file.name.lower()
            if not any(filename.endswith(ext) for ext in ['.xlsx', '.csv', '.txt']):
                return False, "❌ Unsupported file format. Please use .xlsx, .csv, or .txt files."
            
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB limit
                return False, "❌ File too large. Maximum size is 10MB."
            
            return True, "✅ File format validated"
        except Exception as e:
            return False, f"❌ File validation error: {str(e)}"

    @staticmethod
    def validate_data_content(velocity: np.ndarray, absorption: np.ndarray) -> Tuple[bool, str]:
        """Validate data content and structure"""
        try:
            if len(velocity) < 10:
                return False, "❌ Insufficient data points. Minimum 10 points required for reliable fitting."
            
            if len(velocity) != len(absorption):
                return False, "❌ Velocity and absorption arrays have different lengths."
            
            if np.any(np.isnan(velocity)) or np.any(np.isnan(absorption)):
                return False, "❌ Data contains NaN values. Please clean your data."
            
            if np.any(np.isinf(velocity)) or np.any(np.isinf(absorption)):
                return False, "❌ Data contains infinite values. Please check your data."
            
            velocity_range = np.max(velocity) - np.min(velocity)
            if velocity_range < 1.0:
                return False, "❌ Velocity range too small. Expected range > 1 mm/s for typical Mössbauer spectra."
            
            if velocity_range > 50.0:
                return False, "❌ Velocity range unusually large. Please check velocity units (should be mm/s)."
            
            return True, "✅ Data content validated"
        except Exception as e:
            return False, f"❌ Data validation error: {str(e)}"

class PeakDetector:
    @staticmethod
    def detect_peaks(velocity: np.ndarray, absorption: np.ndarray, 
                    min_prominence: float = 0.01, min_distance: int = 5) -> List[int]:
        """Detect peaks in Mössbauer spectrum using prominence and distance criteria"""
        try:
            # Smooth the data to reduce noise
            if len(absorption) > 10:
                smoothed = savgol_filter(absorption, 
                                       window_length=min(11, len(absorption)//3*2+1), 
                                       polyorder=2)
            else:
                smoothed = absorption
            
            # Find peaks (absorption minima are peaks in transmission)
            inverted = 1 - smoothed
            peaks, properties = find_peaks(inverted, 
                                         prominence=min_prominence,
                                         distance=min_distance)
            
            return peaks.tolist()
        except Exception as e:
            st.warning(f"Peak detection failed: {str(e)}")
            return []

    @staticmethod
    def suggest_n_sites(velocity: np.ndarray, absorption: np.ndarray) -> int:
        """Suggest optimal number of sites based on peak detection"""
        peaks = PeakDetector.detect_peaks(velocity, absorption)
        
        if len(peaks) == 0:
            return 1
        elif len(peaks) <= 2:
            return 1  # Single site (doublet)
        elif len(peaks) <= 4:
            return 2  # Two sites
        elif len(peaks) <= 6:
            return 3  # Three sites
        else:
            return min(4, len(peaks) // 2)  # Cap at 4 sites

class MossbauerFitter:
    def __init__(self, model_type: FitModel):
        self.model_type = model_type
        self.velocity = None
        self.absorption = None
        self.result = None
        self.statistics = None
        self.individual_components = {}

    def load_data(self, uploaded_file) -> Tuple[bool, str]:
        """Load and validate data from uploaded file"""
        try:
            # Validate file format
            valid_format, format_message = DataValidator.validate_file_format(uploaded_file)
            if not valid_format:
                return False, format_message

            # Load data based on file type
            if uploaded_file.name.endswith('.xlsx'):
                success, message = self._load_excel_data(uploaded_file)
            elif uploaded_file.name.endswith(('.txt', '.csv')):
                success, message = self._load_text_data(uploaded_file)
            else:
                return False, "❌ Unsupported file format"

            if not success:
                return False, message

            # Validate data content
            valid_content, content_message = DataValidator.validate_data_content(
                self.velocity, self.absorption
            )
            if not valid_content:
                return False, content_message

            # Preprocess data
            self._preprocess_data()

            return True, "✅ Data loaded and preprocessed successfully"

        except Exception as e:
            return False, f"❌ Unexpected error loading data: {str(e)}"

    def _load_excel_data(self, uploaded_file) -> Tuple[bool, str]:
        """Load data from Excel file"""
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            # Remove completely empty rows
            df = df.dropna(how='all').reset_index(drop=True)
            
            if df.empty:
                return False, "❌ Excel file is empty or contains no valid data"
            
            # Clean column names
            df.columns = df.columns.astype(str).str.strip()
            
            # Try to identify velocity and absorption columns
            if df.shape[1] < 2:
                return False, "❌ Excel file must contain at least 2 columns (velocity and absorption)"
            
            # Use first two columns
            velocity_col = df.columns[0]
            absorption_col = df.columns[1]
            
            # Extract data and remove NaN rows
            data_subset = df[[velocity_col, absorption_col]].dropna()
            
            if data_subset.empty:
                return False, "❌ No valid data rows found in Excel file"
            
            self.velocity = data_subset[velocity_col].astype(float).values
            self.absorption = data_subset[absorption_col].astype(float).values
            
            return True, f"✅ Excel data loaded: {len(self.velocity)} data points"
            
        except Exception as e:
            return False, f"❌ Error reading Excel file: {str(e)}"

    def _load_text_data(self, uploaded_file) -> Tuple[bool, str]:
        """Load data from text/CSV file"""
        try:
            content = uploaded_file.read().decode('utf-8')
            uploaded_file.seek(0)
            
            # Try different separators
            separators = ['\t', ',', ' ', ';', '|']
            df = None
            
            for sep in separators:
                try:
                    df = pd.read_csv(io.StringIO(content), sep=sep, header=None, comment='#')
                    if df.shape[1] >= 2 and len(df.dropna()) > 5:
                        break
                except:
                    continue
            
            if df is None or df.shape[1] < 2:
                return False, "❌ Could not parse text file. Ensure data has at least 2 columns separated by tabs, commas, spaces, or semicolons."
            
            # Clean data
            df = df.dropna().reset_index(drop=True)
            
            if df.empty:
                return False, "❌ No valid data found in text file"
            
            # Use first two columns
            self.velocity = df.iloc[:, 0].astype(float).values
            self.absorption = df.iloc[:, 1].astype(float).values
            
            return True, f"✅ Text data loaded: {len(self.velocity)} data points"
            
        except UnicodeDecodeError:
            return False, "❌ File encoding error. Please save your file as UTF-8 text."
        except Exception as e:
            return False, f"❌ Error reading text file: {str(e)}"

    def _preprocess_data(self):
        """Preprocess the loaded data"""
        # Sort data by velocity
        sort_idx = np.argsort(self.velocity)
        self.velocity = self.velocity[sort_idx]
        self.absorption = self.absorption[sort_idx]
        
        # Normalize absorption data
        if self.absorption.max() > 10:
            self.absorption = self.absorption / 100.0
        
        # Ensure absorption is in transmission format (values between 0 and 1)
        if self.absorption.min() < 0:
            self.absorption = self.absorption - self.absorption.min()
        
        if self.absorption.max() > 1:
            self.absorption = self.absorption / self.absorption.max()

    def fit(self, n_sites: int) -> Any:
        """Fit the spectrum with specified number of sites"""
        try:
            # Use peak detection for better initial guesses
            peaks = PeakDetector.detect_peaks(self.velocity, self.absorption)
            
            if len(peaks) >= n_sites:
                # Use detected peaks as initial centers
                centers = self.velocity[peaks[:n_sites]]
            else:
                # Fall back to evenly spaced centers
                centers = np.linspace(self.velocity.min(), self.velocity.max(), n_sites)

            model = None
            self.individual_components = {}
            
            for i in range(n_sites):
                prefix = f"p{i}_"
                
                # Create model based on type
                if self.model_type == FitModel.LORENTZIAN:
                    m = LorentzianModel(prefix=prefix)
                elif self.model_type == FitModel.VOIGT:
                    m = VoigtModel(prefix=prefix)
                else:  # PSEUDO_VOIGT
                    m = PseudoVoigtModel(prefix=prefix)

                model = m if model is None else model + m

            # Create parameters for the entire composite model
            params = model.make_params()
            
            # Debug: Print available parameters
            st.write(f"Debug: Available parameters for {self.model_type.value}: {list(params.keys())}")
            
            # Set parameter values and bounds for each site
            for i in range(n_sites):
                prefix = f"p{i}_"
                
                # Set center parameter
                if f"{prefix}center" in params:
                    params[f"{prefix}center"].set(value=centers[i], 
                                                 min=self.velocity.min(), 
                                                 max=self.velocity.max())
                
                # Set amplitude parameter
                if f"{prefix}amplitude" in params:
                    params[f"{prefix}amplitude"].set(value=0.3, min=0.01, max=2.0)
                
                # Set width/shape parameters based on model type
                if self.model_type == FitModel.LORENTZIAN:
                    if f"{prefix}width" in params:
                        params[f"{prefix}width"].set(value=0.5, min=0.1, max=2.0)
                    else:
                        st.error(f"Parameter {prefix}width not found in Lorentzian model")
                        return None
                        
                elif self.model_type == FitModel.VOIGT:
                    if f"{prefix}sigma" in params:
                        params[f"{prefix}sigma"].set(value=0.3, min=0.05, max=1.5)
                    if f"{prefix}gamma" in params:
                        params[f"{prefix}gamma"].set(value=0.3, min=0.05, max=1.5)
                    
                else:  # PSEUDO_VOIGT
                    if f"{prefix}sigma" in params:
                        params[f"{prefix}sigma"].set(value=0.4, min=0.1, max=2.0)
                    if f"{prefix}fraction" in params:
                        params[f"{prefix}fraction"].set(value=0.5, min=0.0, max=1.0)

            # Perform fit with robust error handling
            self.result = model.fit(self.absorption, params, x=self.velocity, method='leastsq')
            
            # Debug: Print fit result parameters
            st.write(f"Debug: Fit result parameters: {list(self.result.params.keys())}")
            
            # Calculate individual components
            for i in range(n_sites):
                prefix = f"p{i}_"
                
                # Create individual component model
                if self.model_type == FitModel.LORENTZIAN:
                    component_model = LorentzianModel(prefix=prefix)
                elif self.model_type == FitModel.VOIGT:
                    component_model = VoigtModel(prefix=prefix)
                else:
                    component_model = PseudoVoigtModel(prefix=prefix)
                
                # Extract parameters for this component
                component_params = {}
                for param_name, param_value in self.result.params.items():
                    if param_name.startswith(prefix):
                        clean_name = param_name.replace(prefix, '')
                        component_params[clean_name] = param_value.value
                
                # Generate component curve
                try:
                    self.individual_components[f"Site {i+1}"] = component_model.eval(
                        x=self.velocity, **component_params
                    )
                except Exception as comp_error:
                    st.warning(f"Could not generate component {i+1}: {str(comp_error)}")
                    # Create a dummy component to avoid crashes
                    self.individual_components[f"Site {i+1}"] = np.zeros_like(self.velocity)
            
            # Calculate statistics
            self._calculate_statistics()
            
            return self.result
            
        except Exception as e:
            st.error(f"Detailed error in fitting: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Fitting failed: {str(e)}")

    def _calculate_statistics(self):
        """Calculate fit quality statistics"""
        if self.result is None:
            return
        
        try:
            residuals = self.result.residual
            n_data = len(self.absorption)
            n_params = len(self.result.params)
            
            # R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((self.absorption - np.mean(self.absorption))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # RMSE
            rmse = np.sqrt(np.mean(residuals**2))
            
            # Reduced chi-square
            dof = n_data - n_params
            reduced_chi_square = self.result.chisqr / dof if dof > 0 else float('inf')
            
            self.statistics = FitStatistics(
                r_squared=r_squared,
                reduced_chi_square=reduced_chi_square,
                aic=self.result.aic,
                bic=self.result.bic,
                rmse=rmse,
                n_parameters=n_params,
                n_data_points=n_data
            )
            
        except Exception as e:
            st.warning(f"Could not calculate statistics: {str(e)}")
            self.statistics = None

    def get_fit_parameters_table(self) -> pd.DataFrame:
        """Extract fit parameters as a DataFrame"""
        if self.result is None:
            return pd.DataFrame()
        
        data = []
        site_num = 1
        
        for param_name, param in self.result.params.items():
            if param_name.endswith('_center'):
                prefix = param_name.replace('_center', '')
                
                # Extract parameters for this site
                center = param.value
                center_err = param.stderr if param.stderr else 0
                
                # Get amplitude
                amp_param = f"{prefix}_amplitude"
                amplitude = self.result.params[amp_param].value
                amplitude_err = self.result.params[amp_param].stderr if self.result.params[amp_param].stderr else 0
                
                # Get width parameter (depends on model type)
                if self.model_type == FitModel.LORENTZIAN:
                    width_param = f"{prefix}_width"
                    width = self.result.params[width_param].value
                    width_err = self.result.params[width_param].stderr if self.result.params[width_param].stderr else 0
                    width_label = "Width (mm/s)"
                    extra_params = ""
                    
                elif self.model_type == FitModel.VOIGT:
                    sigma_param = f"{prefix}_sigma"
                    gamma_param = f"{prefix}_gamma"
                    sigma = self.result.params[sigma_param].value
                    gamma = self.result.params[gamma_param].value
                    sigma_err = self.result.params[sigma_param].stderr if self.result.params[sigma_param].stderr else 0
                    gamma_err = self.result.params[gamma_param].stderr if self.result.params[gamma_param].stderr else 0
                    
                    # For Voigt, calculate FWHM
                    fwhm = 3.60131 * sigma + np.sqrt(12.25344 * sigma**2 + gamma**2)
                    width = fwhm
                    width_err = 0  # Complex error propagation, simplified
                    width_label = "FWHM (mm/s)"
                    extra_params = f"σ={sigma:.3f}±{sigma_err:.3f}, γ={gamma:.3f}±{gamma_err:.3f}"
                    
                else:  # PSEUDO_VOIGT
                    sigma_param = f"{prefix}_sigma"
                    fraction_param = f"{prefix}_fraction"
                    sigma = self.result.params[sigma_param].value
                    fraction = self.result.params[fraction_param].value
                    sigma_err = self.result.params[sigma_param].stderr if self.result.params[sigma_param].stderr else 0
                    fraction_err = self.result.params[fraction_param].stderr if self.result.params[fraction_param].stderr else 0
                    
                    # For Pseudo-Voigt, width depends on fraction
                    width = sigma * 2  # Approximation
                    width_err = sigma_err * 2
                    width_label = "Width (mm/s)"
                    extra_params = f"σ={sigma:.3f}±{sigma_err:.3f}, η={fraction:.3f}±{fraction_err:.3f}"
                
                # Calculate relative area
                total_area = sum(self.result.params[p].value for p in self.result.params if p.endswith('_amplitude'))
                relative_area = (amplitude / total_area) * 100 if total_area > 0 else 0
                
                # Create row data
                row_data = {
                    'Site': f"Site {site_num}",
                    'Isomer Shift (mm/s)': f"{center:.3f} ± {center_err:.3f}",
                    width_label: f"{width:.3f} ± {width_err:.3f}",
                    'Relative Area (%)': f"{relative_area:.1f}",
                    'Amplitude': f"{amplitude:.3f} ± {amplitude_err:.3f}"
                }
                
                # Add extra parameters for complex models
                if extra_params:
                    row_data['Model Parameters'] = extra_params
                
                data.append(row_data)
                site_num += 1
        
        return pd.DataFrame(data)

class MossbauerInterpreter:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model_name = model_name

    def generate_summary(self, velocity: np.ndarray, absorption: np.ndarray, 
                        fit_model: str, n_sites: int, fit_params: pd.DataFrame,
                        statistics: Optional[FitStatistics] = None) -> str:
        """Generate AI-powered interpretation of the spectrum"""
        try:
            client = anthropic.Anthropic(api_key=self.api_key)

            # Prepare data summary
            values = [f"{v:.3f} {a:.3f}" for v, a in zip(velocity, absorption)]
            data_block = "\n".join(values[:100]) + ("\n..." if len(values) > 100 else "")
            
            # Prepare fit parameters summary
            params_text = fit_params.to_string(index=False) if not fit_params.empty else "No parameters available"
            
            # Prepare statistics summary
            stats_text = ""
            if statistics:
                stats_text = f"""
Fit Quality Statistics:
- R² = {statistics.r_squared:.4f}
- Reduced χ² = {statistics.reduced_chi_square:.4f}
- RMSE = {statistics.rmse:.4f}
- AIC = {statistics.aic:.2f}
- BIC = {statistics.bic:.2f}
"""

            prompt = f"""
You are an expert in Mössbauer spectroscopy analyzing ⁵⁷Fe data. 

SPECTRUM DATA (velocity [mm/s] vs absorption):
{data_block}

FITTING RESULTS:
Model: {fit_model} with {n_sites} site(s)
{params_text}

{stats_text}

Please provide a comprehensive analysis including:

1. **Iron Site Identification**: Based on isomer shift values, identify the likely oxidation states, spin states, and coordination environments

2. **Structural Information**: Discuss what the quadrupole splittings reveal about site symmetry and electronic environments

3. **Fit Quality Assessment**: Evaluate the fit quality using the provided statistics and suggest if the number of sites is appropriate

4. **Mineral/Compound Classification**: Based on the parameters, suggest possible mineral phases or compound types

5. **Recommendations**: Suggest any additional measurements or analysis that would be beneficial

Format your response with clear sections and be specific about the scientific implications.
"""

            message = client.messages.create(
                model=self.model_name,
                max_tokens=4096,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )

            return message.content[0].text.strip()
            
        except Exception as e:
            return f"Error generating AI interpretation: {str(e)}"

def create_plots(fitter: MossbauerFitter) -> go.Figure:
    """Create comprehensive plots including residuals and individual components"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Mössbauer Spectrum', 'Residuals'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Main spectrum plot
    fig.add_trace(
        go.Scatter(
            x=fitter.velocity, 
            y=fitter.absorption,
            mode='lines+markers',
            name='Experimental',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    if fitter.result is not None:
        # Fitted curve
        fig.add_trace(
            go.Scatter(
                x=fitter.velocity,
                y=fitter.result.best_fit,
                mode='lines',
                name='Fitted',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Individual components
        colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray']
        for i, (site_name, component) in enumerate(fitter.individual_components.items()):
            fig.add_trace(
                go.Scatter(
                    x=fitter.velocity,
                    y=component,
                    mode='lines',
                    name=site_name,
                    line=dict(color=colors[i % len(colors)], width=1, dash='dot'),
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Residuals
        fig.add_trace(
            go.Scatter(
                x=fitter.velocity,
                y=fitter.result.residual,
                mode='lines+markers',
                name='Residuals',
                line=dict(color='black', width=1),
                marker=dict(size=3),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Zero line for residuals
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Update layout
    fig.update_xaxes(title_text="Velocity (mm/s)", row=2, col=1)
    fig.update_yaxes(title_text="Absorption", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    
    fig.update_layout(
        height=600,
        title_text="Mössbauer Spectrum Analysis",
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def export_results(fitter: MossbauerFitter, fit_params: pd.DataFrame, 
                  statistics: Optional[FitStatistics], interpretation: str) -> dict:
    """Export results as a comprehensive dictionary"""
    
    export_data = {
        "metadata": {
            "export_date": datetime.now().isoformat(),
            "model_type": fitter.model_type.value,
            "n_sites": len(fitter.individual_components),
            "n_data_points": len(fitter.velocity)
        },
        "raw_data": {
            "velocity": fitter.velocity.tolist(),
            "absorption": fitter.absorption.tolist()
        },
        "fit_results": {
            "parameters": fit_params.to_dict('records'),
            "best_fit": fitter.result.best_fit.tolist() if fitter.result else None,
            "residuals": fitter.result.residual.tolist() if fitter.result else None
        },
        "statistics": asdict(statistics) if statistics else None,
        "interpretation": interpretation,
        "individual_components": {
            name: component.tolist() 
            for name, component in fitter.individual_components.items()
        }
    }
    
    return export_data

def main():
    # Sidebar configuration
    st.sidebar.subheader("🔑 Anthropic API Key")
    api_key_input = st.sidebar.text_input(
        "Enter Claude API key", 
        type="password", 
        value=st.session_state.api_key or "",
        help="Get your API key from https://console.anthropic.com/"
    )
    
    if api_key_input and api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        st.sidebar.success("✅ API key set!")

    st.sidebar.subheader("🧠 Claude Model")
    selected_model = st.sidebar.selectbox(
        "Choose Claude model:",
        options=["claude-sonnet-4-20250514", "claude-opus-4-20250514"],
        index=0,
        help="Sonnet is faster, Opus is more capable"
    )
    st.session_state.claude_model = selected_model

    # Main title
    st.title("⚛️ Enhanced Mössbauer Spectrum Analyzer")
    st.markdown("AI-powered analysis of ⁵⁷Fe Mössbauer spectroscopy data with advanced fitting and interpretation")

    # Fitting options
    st.sidebar.subheader("⚙️ Fitting Options")
    
    selected_fit_model = st.sidebar.selectbox(
        "Fitting model:",
        options=list(FitModel),
        format_func=lambda x: x.value.replace('_', ' ').title(),
        index=0,
        help="Lorentzian: Simple peaks\nVoigt: Gaussian + Lorentzian convolution\nPseudo-Voigt: Weighted sum of Gaussian + Lorentzian"
    )

    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload Mössbauer spectrum file",
        type=["xlsx", "csv", "txt"],
        help="Supported formats: Excel (.xlsx), CSV (.csv), or text (.txt) files with velocity and absorption columns"
    )

    if uploaded_file is not None:
        # Initialize fitter
        fitter = MossbauerFitter(model_type=selected_fit_model)
        
        # Load data
        with st.spinner("Loading and validating data..."):
            success, message = fitter.load_data(uploaded_file)
            
            if not success:
                st.error(message)
                return
            
            st.success(message)

        # Auto-detect number of sites
        suggested_sites = PeakDetector.suggest_n_sites(fitter.velocity, fitter.absorption)
        
        # Peak detection results
        peaks = PeakDetector.detect_peaks(fitter.velocity, fitter.absorption)
        if peaks:
            st.sidebar.info(f"🔍 Detected {len(peaks)} peaks at velocities: {', '.join([f'{fitter.velocity[p]:.2f}' for p in peaks])} mm/s")
        
        n_sites = st.sidebar.number_input(
            "Number of Mössbauer sites:",
            min_value=1,
            max_value=6,
            value=suggested_sites,
            help=f"Auto-detected: {suggested_sites} sites based on peak detection"
        )

        # Fit the data
        try:
            with st.spinner("Fitting spectrum..."):
                result = fitter.fit(n_sites)
                
            st.success("✅ Spectrum fitted successfully!")
            
            # Display comprehensive plots
            fig = create_plots(fitter)
            st.plotly_chart(fig, use_container_width=True)
            
            # Create tabs for different results
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Fit Parameters", "📈 Statistics", "🧠 AI Interpretation", "💾 Export"])
            
            with tab1:
                st.subheader("Fit Parameters")
                fit_params = fitter.get_fit_parameters_table()
                st.dataframe(fit_params, use_container_width=True)
                
                # Download parameters as CSV
                csv_data = fit_params.to_csv(index=False)
                st.download_button(
                    label="📥 Download Parameters (CSV)",
                    data=csv_data,
                    file_name=f"mossbauer_parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with tab2:
                st.subheader("Fit Quality Statistics")
                if fitter.statistics:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("R²", f"{fitter.statistics.r_squared:.4f}")
                        st.metric("RMSE", f"{fitter.statistics.rmse:.4f}")
                        st.metric("Data Points", fitter.statistics.n_data_points)
                    
                    with col2:
                        st.metric("Reduced χ²", f"{fitter.statistics.reduced_chi_square:.4f}")
                        st.metric("AIC", f"{fitter.statistics.aic:.2f}")
                        st.metric("Parameters", fitter.statistics.n_parameters)
                    
                    # Interpretation of statistics
                    st.markdown("### Statistics Interpretation")
                    
                    if fitter.statistics.r_squared > 0.95:
                        st.success("🟢 Excellent fit quality (R² > 0.95)")
                    elif fitter.statistics.r_squared > 0.90:
                        st.info("🟡 Good fit quality (R² > 0.90)")
                    else:
                        st.warning("🟠 Poor fit quality (R² < 0.90) - consider adjusting number of sites")
                    
                    if fitter.statistics.reduced_chi_square < 2:
                        st.success("🟢 Good reduced χ² (< 2)")
                    elif fitter.statistics.reduced_chi_square < 5:
                        st.info("🟡 Acceptable reduced χ² (< 5)")
                    else:
                        st.warning("🟠 High reduced χ² (> 5) - model may be inadequate")
                
                else:
                    st.error("Statistics could not be calculated")
            
            with tab3:
                st.subheader("AI-Powered Interpretation")
                if st.session_state.api_key:
                    with st.spinner("Generating Claude interpretation..."):
                        interpreter = MossbauerInterpreter(
                            api_key=st.session_state.api_key,
                            model_name=st.session_state.claude_model
                        )
                        interpretation = interpreter.generate_summary(
                            fitter.velocity, 
                            fitter.absorption, 
                            selected_fit_model.value.replace('_', ' ').title(), 
                            n_sites,
                            fit_params,
                            fitter.statistics
                        )
                        
                        model_label = "Sonnet" if "sonnet" in st.session_state.claude_model else "Opus"
                        st.markdown(f"**Analysis by Claude 4 {model_label}**")
                        st.markdown(interpretation)
                else:
                    st.warning("🔑 Set your Anthropic API key in the sidebar to enable AI interpretation.")
                    interpretation = "AI interpretation not available - API key not provided"
            
            with tab4:
                st.subheader("Export Results")
                
                # Prepare export data
                export_data = export_results(fitter, fit_params, fitter.statistics, 
                                           interpretation if st.session_state.api_key else "Not available")
                
                # JSON export
                json_data = json.dumps(export_data, indent=2)
                st.download_button(
                    label="📥 Download Complete Analysis (JSON)",
                    data=json_data,
                    file_name=f"mossbauer_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
                
                # Excel export
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # Parameters sheet
                    fit_params.to_excel(writer, sheet_name='Parameters', index=False)
                    
                    # Raw data sheet
                    raw_data_df = pd.DataFrame({
                        'Velocity (mm/s)': fitter.velocity,
                        'Absorption': fitter.absorption,
                        'Fitted': fitter.result.best_fit,
                        'Residuals': fitter.result.residual
                    })
                    raw_data_df.to_excel(writer, sheet_name='Data', index=False)
                    
                    # Statistics sheet
                    if fitter.statistics:
                        stats_df = pd.DataFrame([asdict(fitter.statistics)])
                        stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                
                excel_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download Analysis (Excel)",
                    data=excel_buffer.getvalue(),
                    file_name=f"mossbauer_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Show preview of export data
                st.markdown("### Export Preview")
                st.json(export_data, expanded=False)
            
        except Exception as e:
            st.error(f"❌ Fitting failed: {str(e)}")
            st.info("Try adjusting the number of sites or check your data format.")
    
    else:
        st.info("📁 Please upload a Mössbauer spectrum file to begin analysis.")
        
        # Show example data format
        st.markdown("### Expected Data Format")
        st.markdown("""
        Your file should contain two columns:
        1. **Velocity** (mm/s) - typically ranging from -4 to +4 mm/s
        2. **Absorption** - normalized transmission or absorption values
        
        **Supported formats:**
        - Excel (.xlsx): Any two-column format
        - CSV (.csv): Comma-separated values
        - Text (.txt): Space, tab, or other delimiter-separated values
        """)
        
        # Show example data
        example_data = pd.DataFrame({
            'Velocity (mm/s)': [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            'Absorption': [0.95, 0.94, 0.93, 0.91, 0.88, 0.85, 0.82, 0.79, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.93, 0.94, 0.95]
        })
        
        st.markdown("**Example data:**")
        st.dataframe(example_data, use_container_width=True)

if __name__ == "__main__":
    main()
