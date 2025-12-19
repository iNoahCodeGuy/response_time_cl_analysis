# 🚗 Lead Response Time Analyzer

A Streamlit application that analyzes the impact of response time on lead close rates. Built with rigorous statistical methods and designed to explain complex concepts to non-technical users.

## 🎯 What This App Does

This application helps you answer a critical sales question:

> **"Does responding faster to leads increase the chance of closing a deal?"**

Upload your lead data, and the app will:

1. **Analyze response time patterns** - See how leads are distributed across response time buckets
2. **Calculate close rates** - Compare conversion rates for fast vs. slow responses
3. **Run statistical tests** - Determine if differences are statistically significant
4. **Control for confounders** - Account for lead source and sales rep effects
5. **Explain everything** - Step-by-step explanations for non-technical users

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/iNoahCodeGuy/response_time_cl_analysis.git
cd response_time_cl_analysis

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Try with Sample Data

Don't have data ready? Click **"Load Sample Data"** to generate 10,000 realistic sample leads and explore the app's features.

## 📊 Data Requirements

Your data should include these columns (names can vary):

| Required Data | Example Column Names |
|--------------|---------------------|
| Lead arrival time | `created_at`, `lead_time`, `timestamp` |
| First response time | `first_response`, `replied_at`, `contacted_at` |
| Lead source | `source`, `channel`, `lead_source` |
| Sales rep | `rep`, `agent`, `salesperson` |
| Order outcome | `ordered`, `sold`, `converted` |

### Supported Formats

- **File types**: CSV, XLSX, XLS
- **Date formats**: ISO, US, European, Excel serial numbers
- **Outcome values**: Boolean, 1/0, Yes/No, text values like "Ordered"

## 🔬 Analysis Features

### Standard Analysis Mode
- **Descriptive statistics** - Close rates by response time bucket
- **Chi-square test** - Is there any relationship?
- **Z-test for proportions** - Pairwise bucket comparisons
- **Logistic regression** - Effect size controlling for lead source

### Advanced Analysis Mode (Additional)
- **Mixed effects model** - Control for sales rep random effects
- **Within-rep analysis** - Compare fast vs. slow within same rep
- **Confounding assessment** - Evaluate potential bias in results

## 📈 Understanding the Results

### Key Metrics

| Metric | What It Means |
|--------|---------------|
| **Close Rate** | Percentage of leads that converted to orders |
| **P-Value** | Probability the result is due to chance (< 0.05 = significant) |
| **Odds Ratio** | How much higher/lower the odds of ordering are vs. reference |
| **Confidence Interval** | Range where the true value likely falls |

### Interpreting Significance

- **Significant (p < 0.05)**: Differences unlikely due to random chance
- **Not Significant (p ≥ 0.05)**: Cannot rule out random variation

## 🏗️ Project Structure

```
response_time_cl_analysis/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── config/
│   └── settings.py             # Configuration constants
│
├── data/
│   ├── loader.py               # File loading and validation
│   ├── datetime_parser.py      # Date/time format detection
│   ├── column_mapper.py        # Column mapping logic
│   └── sample_generator.py     # Sample data generation
│
├── analysis/
│   ├── preprocessing.py        # Response time bucketing
│   ├── descriptive.py          # Summary statistics
│   ├── statistical_tests.py    # Chi-square, z-tests
│   ├── regression.py           # Logistic regression
│   └── advanced.py             # Mixed effects, within-rep analysis
│
├── explanations/
│   ├── templates.py            # Plain-English explanations
│   └── formulas.py             # LaTeX formulas
│
└── components/
    ├── upload.py               # File upload interface
    ├── mapping_ui.py           # Column mapping interface
    ├── settings_panel.py       # Settings sidebar
    ├── step_display.py         # Step-by-step explanations
    ├── charts.py               # Plotly visualizations
    └── results_dashboard.py    # Results display
```

## ⚠️ Limitations

This analysis has important limitations to keep in mind:

1. **Correlation ≠ Causation** - This is observational data, not a controlled experiment
2. **Unmeasured Confounders** - We can only control for variables we measure
3. **Selection Bias** - Only analyzes leads that received responses
4. **External Validity** - Results may not generalize to different contexts

**For causal conclusions**, consider running an A/B test where response times are randomly varied.

## 🛠️ Customization

### Response Time Buckets

Edit `config/settings.py` to change default bucket boundaries:

```python
DEFAULT_BUCKETS = [0, 15, 30, 60, float('inf')]
DEFAULT_BUCKET_LABELS = ['0-15 min', '15-30 min', '30-60 min', '60+ min']
```

### Significance Level

Change the default alpha level in `config/settings.py`:

```python
DEFAULT_ALPHA = 0.05  # Change to 0.01 for stricter threshold
```

## 📚 Statistical Methods

### Chi-Square Test of Independence

Tests whether response time bucket and order outcome are independent:

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

### Z-Test for Proportions

Compares close rates between two specific buckets:

$$z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}$$

### Logistic Regression

Models log-odds of ordering as a function of response time and controls:

$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 \cdot \text{bucket} + \beta_2 \cdot \text{source}$$

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional statistical tests
- More visualization options
- Enhanced confounding diagnostics
- A/B test power calculator

## 📝 License

This project is for internal use. Modify and distribute as needed for your organization.

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - App framework
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Statsmodels](https://www.statsmodels.org/) - Statistical models
- [SciPy](https://scipy.org/) - Statistical tests

---

*"Does response time matter? Now you have the data to know for sure."*
