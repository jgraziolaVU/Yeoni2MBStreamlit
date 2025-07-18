# Mössbauer Spectrum Analyzer (Streamlit Version)

A streamlined web application for analyzing ⁵⁷Fe Mössbauer spectroscopy data with AI-powered interpretation using Claude 4 Sonnet.

## Features

- **Automated Spectrum Fitting**: Intelligent peak detection and fitting using Lorentzian, Voigt, or Pseudo-Voigt models
- **AI-Powered Analysis**: Claude 4 Sonnet powered interpretation of iron sites, oxidation states, and coordination environments
- **Interactive Visualization**: Real-time plotting with Plotly
- **Multiple File Formats**: Support for .txt, .csv, and .xlsx files
- **Simple Deployment**: Single Python file, easy to deploy on Streamlit Community Cloud
- **Export Capabilities**: Download detailed analysis reports in JSON format

## Quick Start

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mossbauer-analyzer.git
cd mossbauer-analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run app.py
```

5. Open your browser to `http://localhost:8501`

### Deployment on Streamlit Community Cloud

1. Fork this repository
2. Sign up for [Streamlit Community Cloud](https://share.streamlit.io/)
3. Deploy directly from your GitHub repository
4. Your app will be available at `https://share.streamlit.io/[username]/[repo-name]`

## Usage

1. **Set API Key**: Enter your Anthropic API key in the sidebar (get one at https://console.anthropic.com/)
2. **Configure Options**: Adjust fitting model, number of sites, or baseline correction in the sidebar
3. **Upload Data**: Drag and drop or click to upload your Mössbauer spectrum file
4. **Review Results**: 
   - Interactive plot showing experimental data and fitted curves
   - Residuals plot for fit quality assessment
   - Parameter table with isomer shifts, quadrupole splittings, and site assignments
   - AI-generated interpretation using Claude 4 Sonnet
5. **Export**: Download the complete analysis report as JSON

## File Format

Input files should contain two columns:
- Column 1: Velocity (mm/s)
- Column 2: Absorption or transmission (normalized or percentage)

Supported delimiters: space, tab, comma, semicolon

## Example Data

You can test the app with sample Mössbauer data files available in the `examples/` directory.

## For Scientists

This tool is designed for research laboratories and scientists working with Mössbauer spectroscopy. Key benefits:

- **No installation required**: Access via web browser
- **Reproducible results**: All fitting parameters are saved
- **Publication-ready**: Export detailed reports for papers
- **Collaborative**: Share results with colleagues easily
- **Extensible**: Python code can be modified for specific needs

## Technical Details

The app uses:
- **lmfit** for non-linear least-squares fitting
- **Plotly** for interactive visualizations
- **Anthropic Claude 4 Sonnet** for AI interpretation
- **Streamlit** for the web interface

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the GPL-3.0 License.

## Acknowledgments

- Mössbauer spectroscopy community for domain knowledge
- Anthropic for Claude 4 Sonnet API
- Streamlit for the excellent web framework
