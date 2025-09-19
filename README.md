# Data Quality Toolkit

A comprehensive data quality validation toolkit built with Python and Streamlit, designed to help you analyze and ensure the quality of your datasets.

## Features

- **Interactive Web Interface**: Streamlit-based UI for easy data quality analysis
- **Comprehensive Quality Checks**: Multiple validation rules including:
  - Unique value validation
  - Not null constraints
  - Data type validation
  - Custom validation rules
- **Visual Reports**: Interactive charts and tables for quality metrics
- **Historical Tracking**: Track data quality over time
- **Export Capabilities**: Generate reports and export results

## Files Overview

- `data_quality_checker.py` - Core data quality validation engine
- `data_quality_streamlit_app.py` - Streamlit web application
- `run_streamlit_app.py` - Launch script for the application
- `requirements_streamlit.txt` - Python dependencies
- `test_data.csv` - Sample dataset for testing
- `sample_historical_tracking.csv` - Example historical tracking data

## Installation

1. Clone this repository:
```bash
git clone https://github.com/drewdezco/data_quality_streamlit.git
cd data_quality_streamlit
```

2. Install dependencies:
```bash
pip install -r requirements_streamlit.txt
```

## Usage

### Quick Start
Run the application using the launch script:
```bash
python run_streamlit_app.py
```

### Manual Start
Alternatively, start Streamlit directly:
```bash
streamlit run data_quality_streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## How to Use

1. **Upload Data**: Use the file uploader to select your CSV file
2. **Configure Rules**: Set up validation rules for your dataset
3. **Run Analysis**: Execute the data quality checks
4. **Review Results**: Examine the detailed quality report
5. **Track Progress**: Monitor data quality improvements over time

## Requirements

See `requirements_streamlit.txt` for a complete list of Python dependencies.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.
