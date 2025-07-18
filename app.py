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

def find_peaks(
    data: np.ndarray, height: float = 0.0, distance: int = 1
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Simple peak finding replacement for :func:`scipy.signal.find_peaks`.

    Parameters
    ----------
    data : ndarray
        1-D array in which to find local maxima.
    height : float, optional
        Minimum peak height.
    distance : int, optional
        Minimum number of samples between adjacent peaks.

    Returns
    -------
    peaks : ndarray
        Indices of the peaks in ``data``.
    properties : dict
        Dictionary containing the ``peak_heights`` array.
    """
    data = np.asarray(data)
    if len(data) < 3:
        return np.array([], dtype=int), {"peak_heights": np.array([])}

    peaks: List[int] = []
    for i in range(1, len(data) - 1):
        if data[i] > data[i - 1] and data[i] > data[i + 1] and data[i] >= height:
            if peaks and i - peaks[-1] < distance:
                # Keep the higher of two close peaks
                if data[i] > data[peaks[-1]]:
                    peaks[-1] = i
            else:
                peaks.append(i)

    peak_indices = np.array(peaks, dtype=int)
    return peak_indices, {"peak_heights": data[peak_indices] if len(peaks) else np.array([])}

# Page configuration
st.set_page_config(
    page_title="Mössbauer Spectrum Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stButton > button {
        width: 100%;
    }
    .plot-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

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
    """Represents a single Mössbauer site with all relevant parameters"""
    isomer_shift: float
    quadrupole_splitting: float
    line_width: float
    relative_area: float
    site_type: str = ""
    hyperfine_field: Optional[float] = None
    
    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class InterpretationRule:
    """Rule-based interpretation for when AI is not available"""
    condition: callable
    interpretation: str

class MossbauerFitter:
    def __init__(self, model_type: FitModel = FitModel.LORENTZIAN):
        self.model_type = model_type
        self.velocity = None
        self.absorption = None
        self.result = None
        
    def load_data(self, uploaded_file, apply_baseline_correction: bool = True) -> Tuple[bool, str]:
        """Load and validate data from file

        Parameters
        ----------
        uploaded_file: Uploaded file-like object
            The spectrum data file provided by the user.
        apply_baseline_correction: bool, optional
            Whether to normalize the absorption data using a simple baseline
            correction. Defaults to ``True``.
        """
        try:
            # Read file based on type
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.txt') or uploaded_file.name.endswith('.csv'):
                # Try multiple delimiters
                content = uploaded_file.read().decode('utf-8')
                uploaded_file.seek(0)  # Reset file pointer
                
                for sep in ['\t', ' ', ',', ';']:
                    try:
                        df = pd.read_csv(io.StringIO(content), sep=sep, header=None)
                        if df.shape[1] >= 2:
                            break
                    except:
                        continue
            else:
                return False, "Unsupported file format. Please use .xlsx, .txt, or .csv"
            
            # Validate data
            df = df.dropna().reset_index(drop=True)
            if df.shape[1] < 2:
                return False, "File must contain at least 2 columns (velocity and absorption)"
            
            if df.shape[0] < 10:
                return False, "Insufficient data points (minimum 10 required)"
            
            self.velocity = df.iloc[:, 0].values.astype(float)
            self.absorption = df.iloc[:, 1].values.astype(float)
            
            # Normalize absorption to percentage if needed
            if self.absorption.max() > 10:
                self.absorption = self.absorption / 100.0
            
            # Apply baseline correction if requested
            if apply_baseline_correction and self.absorption.min() < 0.9:
                baseline = np.percentile(self.absorption, 95)
                if baseline:
                    self.absorption = self.absorption / baseline
            
            return True, "Data loaded successfully"
            
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def detect_number_of_sites(self) -> int:
        """Auto-detect the likely number of sites using peak finding"""
        # Invert for peak finding (absorption dips are peaks when inverted)
        inverted = -self.absorption
        peaks, properties = find_peaks(inverted, 
                                     height=np.std(inverted) * 0.5,
                                     distance=len(self.velocity) // 20)
        
        # Mössbauer sites often appear as doublets due to quadrupole splitting
        # So actual number of sites is roughly half the number of peaks
        estimated_sites = max(1, len(peaks) // 2)
        return min(estimated_sites, 4)  # Cap at 4 for practical reasons
    
    def create_model(self, n_sites: Optional[int] = None) -> Tuple[Any, Parameters]:
        """Create the fitting model with appropriate number of sites"""
        if n_sites is None:
            n_sites = self.detect_number_of_sites()
        
        # Select model class based on type
        model_class = {
            FitModel.LORENTZIAN: LorentzianModel,
            FitModel.VOIGT: VoigtModel,
            FitModel.PSEUDO_VOIGT: PseudoVoigtModel
        }[self.model_type]
        
        # Build composite model
        model = None
        for i in range(n_sites * 2):  # Double for doublets
            prefix = f'peak{i+1}_'
            if model is None:
                model = model_class(prefix=prefix)
            else:
                model += model_class(prefix=prefix)
        
        # Create parameters with intelligent initial guesses
        params = model.make_params()
        
        # Find approximate peak positions
        inverted = -self.absorption
        peaks, _ = find_peaks(inverted, height=np.std(inverted) * 0.3)
        peak_velocities = self.velocity[peaks] if len(peaks) > 0 else []
        
        for i in range(n_sites * 2):
            prefix = f'peak{i+1}_'
            
            # Initial center positions
            if i < len(peak_velocities):
                center = peak_velocities[i]
            else:
                # Distribute evenly if no peaks found
                v_min, v_max = self.velocity.min(), self.velocity.max()
                center = v_min + (i + 0.5) * (v_max - v_min) / (n_sites * 2)
            
            params[f'{prefix}center'].set(value=center, 
                                        min=self.velocity.min(), 
                                        max=self.velocity.max())
            
            # Amplitude based on absorption depth
            amplitude = np.ptp(self.absorption) / (n_sites * 2)
            params[f'{prefix}amplitude'].set(value=amplitude, min=0)
            
            # Reasonable line width
            params[f'{prefix}sigma'].set(value=0.15, min=0.05, max=1.0)
            
            # For Voigt model, set gamma
            if self.model_type == FitModel.VOIGT:
                params[f'{prefix}gamma'].set(value=0.15, min=0.05, max=1.0)
        
        return model, params
    
    def fit(self, n_sites: Optional[int] = None) -> Dict[str, Any]:
        """Perform the fitting"""
        model, params = self.create_model(n_sites)
        
        # Perform the fit with weights
        weights = 1.0 / np.sqrt(self.absorption + 0.001)  # Poisson-like weights
        self.result = model.fit(self.absorption, params, x=self.velocity, weights=weights)
        
        # Extract Mössbauer parameters
        sites = self._extract_mossbauer_parameters(n_sites)
        
        return {
            "success": True,
            "sites": sites,
            "chi_squared": float(self.result.chisqr),
            "reduced_chi_squared": float(self.result.redchi),
            "n_data_points": len(self.velocity),
            "n_variables": self.result.nvarys,
            "fit_report": self.result.fit_report()
        }
    
    def _extract_mossbauer_parameters(self, n_sites: Optional[int] = None) -> List[MossbauerSite]:
        """Extract parameters in Mössbauer-relevant format"""
        if n_sites is None:
            n_sites = self.detect_number_of_sites()
        
        sites = []
        components = self.result.eval_components()
        
        # Group peaks into sites (pairs for doublets)
        for site_idx in range(n_sites):
            peak1_name = f'peak{site_idx*2 + 1}_'
            peak2_name = f'peak{site_idx*2 + 2}_'
            
            if peak1_name in components and peak2_name in components:
                # Extract parameters for doublet
                center1 = self.result.params[f'{peak1_name}center'].value
                center2 = self.result.params[f'{peak2_name}center'].value
                
                # Mössbauer parameters
                isomer_shift = (center1 + center2) / 2
                quadrupole_splitting = abs(center2 - center1)
                
                # Average line width
                sigma1 = self.result.params[f'{peak1_name}sigma'].value
                sigma2 = self.result.params[f'{peak2_name}sigma'].value
                line_width = 2 * (sigma1 + sigma2) / 2  # Convert to FWHM
                
                # Calculate relative area
                area1 = np.trapz(components[peak1_name], self.velocity)
                area2 = np.trapz(components[peak2_name], self.velocity)
                total_area = sum(np.trapz(comp, self.velocity) for comp in components.values())
                relative_area = ((area1 + area2) / total_area) * 100 if total_area > 0 else 0
                
                # Determine site type based on parameters
                site_type = self._identify_site_type(isomer_shift, quadrupole_splitting)
                
                site = MossbauerSite(
                    isomer_shift=float(isomer_shift),
                    quadrupole_splitting=float(quadrupole_splitting),
                    line_width=float(line_width),
                    relative_area=float(relative_area),
                    site_type=site_type
                )
                sites.append(site)
        
        return sites
    
    def _identify_site_type(self, isomer_shift: float, quadrupole_splitting: float) -> str:
        """Identify iron site type based on Mössbauer parameters"""
        # Simplified identification based on common ranges
        if -0.2 <= isomer_shift <= 0.5:
            if quadrupole_splitting < 0.5:
                return "Fe³⁺ (low-spin)"
            else:
                return "Fe³⁺ (high-spin)"
        elif 0.6 <= isomer_shift <= 1.5:
            if quadrupole_splitting < 1.0:
                return "Fe²⁺ (low-spin)"
            else:
                return "Fe²⁺ (high-spin)"
        else:
            return "Unknown"
    
    def get_plot_data(self) -> Dict[str, Any]:
        """Generate Plotly plot data"""
        if not self.result:
            return {}
        
        components = self.result.eval_components()
        
        # Create traces
        traces = [
            go.Scatter(
                x=self.velocity.tolist(), 
                y=self.absorption.tolist(), 
                mode='markers',
                name='Experimental',
                marker=dict(size=4, color='black'),
                showlegend=True
            ),
            go.Scatter(
                x=self.velocity.tolist(), 
                y=self.result.best_fit.tolist(), 
                mode='lines',
                name='Total Fit',
                line=dict(color='red', width=2),
                showlegend=True
            )
        ]
        
        # Add component traces
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        for i, (name, component) in enumerate(components.items()):
            traces.append(go.Scatter(
                x=self.velocity.tolist(),
                y=component.tolist(),
                mode='lines',
                name=name.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=1.5, dash='dash'),
                showlegend=True
            ))
        
        # Create residuals
        residuals = self.absorption - self.result.best_fit
        
        return {
            "traces": traces,
            "residuals": residuals
        }

class MossbauerInterpreter:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.use_ai = api_key is not None
        self.rules = self._create_interpretation_rules()
    
    def _create_interpretation_rules(self) -> List[InterpretationRule]:
        """Create rule-based interpretations for common Mössbauer patterns"""
        return [
            InterpretationRule(
                condition=lambda site: -0.2 <= site['isomer_shift'] <= 0.5 and site['quadrupole_splitting'] < 0.5,
                interpretation="Low-spin Fe³⁺ in octahedral coordination"
            ),
            InterpretationRule(
                condition=lambda site: -0.2 <= site['isomer_shift'] <= 0.5 and 0.5 <= site['quadrupole_splitting'] <= 2.0,
                interpretation="High-spin Fe³⁺ in octahedral coordination with distortion"
            ),
            InterpretationRule(
                condition=lambda site: 0.6 <= site['isomer_shift'] <= 0.9 and site['quadrupole_splitting'] < 1.0,
                interpretation="Low-spin Fe²⁺ in octahedral coordination"
            ),
            InterpretationRule(
                condition=lambda site: 0.9 <= site['isomer_shift'] <= 1.5 and site['quadrupole_splitting'] > 1.5,
                interpretation="High-spin Fe²⁺ in octahedral coordination"
            ),
            InterpretationRule(
                condition=lambda site: site['isomer_shift'] > 1.5 and site['quadrupole_splitting'] > 2.5,
                interpretation="Fe²⁺ in tetrahedral coordination"
            ),
        ]
    
    def generate_summary(self, fit_results: Dict[str, Any]) -> str:
        """Generate interpretation summary"""
        if self.use_ai:
            try:
                return self._generate_ai_summary(fit_results)
            except Exception as e:
                st.warning(f"AI summary failed: {e}. Using rule-based interpretation.")
                return self._generate_rule_based_summary(fit_results)
        else:
            return self._generate_rule_based_summary(fit_results)
    
    def _generate_ai_summary(self, fit_results: Dict[str, Any]) -> str:
        """Generate AI-powered interpretation using Claude"""
        client = anthropic.Anthropic(api_key=self.api_key)
        sites = fit_results.get('sites', [])
        
        # Prepare detailed prompt
        sites_description = []
        for i, site in enumerate(sites, 1):
            sites_description.append(
                f"Site {i}: IS = {site.isomer_shift:.3f} mm/s, "
                f"QS = {site.quadrupole_splitting:.3f} mm/s, "
                f"LW = {site.line_width:.3f} mm/s, "
                f"Area = {site.relative_area:.1f}%"
            )
        
        prompt = f"""You are an expert in Mössbauer spectroscopy. Analyze the following ⁵⁷Fe Mössbauer spectrum fitting results and provide a detailed interpretation.

Fitting Results:
{chr(10).join(sites_description)}

Chi-squared: {fit_results.get('chi_squared', 0):.4f}
Reduced chi-squared: {fit_results.get('reduced_chi_squared', 0):.4f}

Please provide:
1. Identification of each iron site (oxidation state, spin state, coordination)
2. Possible mineral phases or compounds
3. Quality assessment of the fit
4. Any notable features or concerns

Write your response as a concise paragraph suitable for a research paper."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text.strip()
    
    def _generate_rule_based_summary(self, fit_results: Dict[str, Any]) -> str:
        """Generate rule-based interpretation when AI is not available"""
        sites = fit_results.get('sites', [])
        interpretations = []
        
        for i, site in enumerate(sites, 1):
            # Find matching interpretation
            site_interpretation = "Unknown iron environment"
            site_dict = site.to_dict() if hasattr(site, 'to_dict') else site
            
            for rule in self.rules:
                if rule.condition(site_dict):
                    site_interpretation = rule.interpretation
                    break
            
            interpretations.append(
                f"Site {i} ({site_dict['relative_area']:.1f}%): {site_interpretation} "
                f"(IS={site_dict['isomer_shift']:.2f}, QS={site_dict['quadrupole_splitting']:.2f} mm/s)"
            )
        
        # Add fit quality assessment
        chi_squared = fit_results.get('reduced_chi_squared', 0)
        if chi_squared < 1.5:
            quality = "excellent"
        elif chi_squared < 3.0:
            quality = "good"
        else:
            quality = "moderate"
        
        summary = f"The spectrum shows {len(sites)} distinct iron site(s). "
        summary += " ".join(interpretations)
        summary += f" The fit quality is {quality} with a reduced χ² of {chi_squared:.3f}."
        
        return summary

def main():
    st.title("⚛️ Mössbauer Spectrum Analyzer")
    st.markdown("AI-powered analysis of ⁵⁷Fe Mössbauer spectroscopy data")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        st.subheader("🔑 Anthropic API Key")
        api_key_input = st.text_input(
            "Enter your API key",
            type="password",
            value=st.session_state.api_key or "",
            help="Required for AI-powered interpretation using Claude 4 Sonnet"
        )
        
        if api_key_input and api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            st.success("API key set successfully!")
        
        st.divider()
        
        # Fitting options
        st.subheader("📊 Fitting Options")
        
        model_type = st.selectbox(
            "Fitting Model",
            options=[FitModel.LORENTZIAN, FitModel.VOIGT, FitModel.PSEUDO_VOIGT],
            format_func=lambda x: x.value.replace('_', ' ').title(),
            help="Lorentzian is suitable for most Mössbauer spectra"
        )
        
        n_sites = st.number_input(
            "Number of Iron Sites",
            min_value=1,
            max_value=6,
            value=None,
            help="Leave empty for automatic detection",
            placeholder="Auto-detect"
        )
        
        baseline_correction = st.checkbox(
            "Apply baseline correction",
            value=True,
            help="Normalizes spectrum to improve fitting"
        )
        
        st.divider()
        
        # About section
        st.subheader("ℹ️ About")
        st.info(
            "This app analyzes ⁵⁷Fe Mössbauer spectra using automated fitting "
            "and AI-powered interpretation. Upload your spectrum file to begin."
        )
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a spectrum file",
            type=['txt', 'csv', 'xlsx'],
            help="File should contain velocity (mm/s) and absorption/transmission data"
        )
        
        if uploaded_file is not None:
            # Process the file
            with st.spinner("Analyzing spectrum..."):
                # Initialize fitter
                fitter = MossbauerFitter(model_type=model_type)
                
                # Load data
                success, message = fitter.load_data(
                    uploaded_file, apply_baseline_correction=baseline_correction
                )
                
                if success:
                    st.success(message)
                    
                    # Perform fitting
                    try:
                        fit_results = fitter.fit(n_sites=n_sites)
                        st.session_state.fit_results = fit_results
                        
                        # Get plot data
                        plot_data = fitter.get_plot_data()
                        st.session_state.results = plot_data
                        
                        # Show fit quality metrics
                        st.subheader("📈 Fit Quality")
                        col_metric1, col_metric2 = st.columns(2)
                        with col_metric1:
                            st.metric("χ²", f"{fit_results['chi_squared']:.4f}")
                        with col_metric2:
                            st.metric("Reduced χ²", f"{fit_results['reduced_chi_squared']:.4f}")
                        
                        # Parameter table
                        st.subheader("🔬 Fitted Parameters")
                        sites_data = []
                        for i, site in enumerate(fit_results['sites'], 1):
                            site_dict = site.to_dict() if hasattr(site, 'to_dict') else site
                            sites_data.append({
                                "Site": i,
                                "IS (mm/s)": f"{site_dict['isomer_shift']:.3f}",
                                "QS (mm/s)": f"{site_dict['quadrupole_splitting']:.3f}",
                                "LW (mm/s)": f"{site_dict['line_width']:.3f}",
                                "Area (%)": f"{site_dict['relative_area']:.1f}",
                                "Assignment": site_dict['site_type']
                            })
                        
                        df_sites = pd.DataFrame(sites_data)
                        st.dataframe(df_sites, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Fitting failed: {str(e)}")
                else:
                    st.error(message)
    
    with col2:
        if st.session_state.results is not None:
            st.header("📊 Results")
            
            # Main spectrum plot
            fig_main = go.Figure(data=st.session_state.results['traces'])
            fig_main.update_layout(
                title="Mössbauer Spectrum Analysis",
                xaxis_title="Velocity (mm/s)",
                yaxis_title="Relative Transmission",
                height=500,
                hovermode='x unified',
                showlegend=True,
                legend=dict(x=1.02, y=1, xanchor="left", yanchor="top")
            )
            st.plotly_chart(fig_main, use_container_width=True)
            
            # Residuals plot
            residuals = st.session_state.results['residuals']
            fig_residuals = go.Figure(data=[
                go.Scatter(
                    x=list(range(len(residuals))),
                    y=residuals.tolist(),
                    mode='markers',
                    marker=dict(size=3, color='gray'),
                    name='Residuals'
                )
            ])
            fig_residuals.update_layout(
                title="Fit Residuals",
                xaxis_title="Data Point",
                yaxis_title="Residuals",
                height=200,
                showlegend=False
            )
            st.plotly_chart(fig_residuals, use_container_width=True)
            
            # AI Interpretation
            if st.session_state.fit_results is not None:
                st.header("🧠 AI Interpretation")
                
                interpreter = MossbauerInterpreter(api_key=st.session_state.api_key)
                with st.spinner("Generating interpretation..."):
                    summary = interpreter.generate_summary(st.session_state.fit_results)
                
                st.info(summary)
                
                # Export options
                st.header("💾 Export Results")
                
                # Prepare report data
                report = {
                    "filename": uploaded_file.name,
                    "analysis_date": datetime.now().isoformat(),
                    "fit_parameters": {
                        "model": model_type.value,
                        "n_sites": len(st.session_state.fit_results['sites']),
                        "n_data_points": st.session_state.fit_results['n_data_points'],
                        "n_variables": st.session_state.fit_results['n_variables']
                    },
                    "sites": [site.to_dict() if hasattr(site, 'to_dict') else site 
                             for site in st.session_state.fit_results['sites']],
                    "fit_quality": {
                        "chi_squared": st.session_state.fit_results['chi_squared'],
                        "reduced_chi_squared": st.session_state.fit_results['reduced_chi_squared']
                    },
                    "interpretation": summary
                }
                
                # Download button
                st.download_button(
                    label="📥 Download Report (JSON)",
                    data=json.dumps(report, indent=2),
                    file_name=f"mossbauer_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        else:
            st.info("👈 Upload a spectrum file to begin analysis")

if __name__ == "__main__":
    main()
