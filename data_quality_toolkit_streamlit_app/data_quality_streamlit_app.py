import streamlit as st
import pandas as pd
import io
import base64
from datetime import datetime
import json
import re
from data_quality_checker import DataQualityChecker

# Configure Streamlit page
st.set_page_config(
    page_title="Data Quality Toolkit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #111827 0%, #1f2937 50%, #374151 100%);
        color: #e5e7eb;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        color: #e0e6ed;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #60a5fa;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .feature-card h3 {
        color: #60a5fa;
        margin-top: 0;
    }
    
    .feature-card ul {
        color: #e0e6ed;
        margin-bottom: 0;
    }
    
    .feature-card li {
        color: #e0e6ed;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        color: #e0e6ed;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .metric-card h3 {
        color: #60a5fa;
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .metric-card p {
        color: #9ca3af;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    
    .success-message {
        background: rgba(34, 197, 94, 0.1);
        color: #4ade80;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #22c55e;
        margin: 1rem 0;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .warning-message {
        background: rgba(245, 158, 11, 0.1);
        color: #fbbf24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #f59e0b;
        margin: 1rem 0;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .step-container {
        background: rgba(255, 255, 255, 0.05);
        color: #e0e6ed;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .step-container h3 {
        color: #60a5fa;
        margin-top: 0;
    }
    
    .step-container p {
        color: #e0e6ed;
    }
    
    /* Style for markdown text in cards */
    .feature-card strong {
        color: #60a5fa;
    }
    
    /* Tab styling for better visibility in dark mode */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0;
        border-radius: 0;
        border: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #9ca3af;
        background-color: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0.75rem 1.5rem !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #e5e7eb !important;
        border-bottom: 2px solid rgba(96, 165, 250, 0.5) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #60a5fa !important;
        border-bottom: 2px solid #60a5fa !important;
        font-weight: 600 !important;
    }
    
    /* Improve text readability in dataframes */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
    
    /* Better sidebar contrast */
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.02);
    }
    
    /* Improve multiselect dropdown readability */
    .stMultiSelect > div > div {
        background-color: rgba(255, 255, 255, 0.05);
        color: #e0e6ed;
    }
    
    /* Better button styling */
    .stButton > button {
        background-color: #1e40af;
        color: white;
        border: none;
        border-radius: 6px;
    }
    
    .stButton > button:hover {
        background-color: #3b82f6;
        color: white;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background-color: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(96, 165, 250, 0.5);
        border-radius: 8px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.05);
        color: #e0e6ed;
    }
    
    /* Hide sidebar completely */
    .css-1d391kg {
        display: none;
    }
    
    /* Adjust main content to use full width */
    .main .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: none;
    }
</style>
""", unsafe_allow_html=True)

def create_enhanced_dataset(df, checker, validation_config):
    """
    Create an enhanced dataset with validation pass/fail columns appended.
    
    Args:
        df: Original dataframe
        checker: DataQualityChecker instance
        validation_config: Validation configuration from session state
    
    Returns:
        Enhanced dataframe with validation result columns
    """
    enhanced_df = df.copy()
    
    # Add validation result columns
    config = validation_config
    
    # Not null validations
    for col in config.get('not_null', []):
        if col in df.columns:
            enhanced_df[f'{col}_not_null_pass'] = ~df[col].isnull()
    
    # Uniqueness validations (check for duplicates)
    for col in config.get('unique', []):
        if col in df.columns:
            enhanced_df[f'{col}_unique_pass'] = ~df[col].duplicated(keep=False) | df[col].isnull()
    
    # Range validations
    for col, (min_val, max_val) in config.get('ranges', {}).items():
        if col in df.columns:
            try:
                enhanced_df[f'{col}_range_pass'] = (
                    (df[col] >= min_val) & (df[col] <= max_val)
                ) | df[col].isnull()  # Null values don't fail range checks
            except:
                enhanced_df[f'{col}_range_pass'] = True  # If comparison fails, mark as pass
    
    # Allowed values validations
    for col, allowed_vals in config.get('allowed_values', {}).items():
        if col in df.columns and allowed_vals:
            enhanced_df[f'{col}_allowed_values_pass'] = (
                df[col].isin(allowed_vals)
            ) | df[col].isnull()
    
    # Pattern validations
    for col, pattern in config.get('patterns', {}).items():
        if col in df.columns and pattern:
            try:
                enhanced_df[f'{col}_pattern_pass'] = (
                    df[col].astype(str).str.match(pattern, na=False)
                ) | df[col].isnull()
            except:
                enhanced_df[f'{col}_pattern_pass'] = True  # If regex fails, mark as pass
    
    # Add overall validation summary for each row
    validation_columns = [col for col in enhanced_df.columns if col.endswith('_pass')]
    if validation_columns:
        enhanced_df['total_validations'] = len(validation_columns)
        enhanced_df['passed_validations'] = enhanced_df[validation_columns].sum(axis=1)
        enhanced_df['validation_success_rate'] = (
            enhanced_df['passed_validations'] / enhanced_df['total_validations'] * 100
        ).round(1)
        enhanced_df['all_validations_pass'] = enhanced_df['passed_validations'] == enhanced_df['total_validations']
    
    return enhanced_df

def calculate_overall_data_quality(df):
    """
    Calculate overall data quality metrics for the dataset.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with completeness, uniqueness, consistency scores and type distribution
    """
    def classify_data_type(col_data):
        """Classify data type with user-friendly names"""
        try:
            dtype_str = str(col_data.dtype).lower()
            if 'object' in dtype_str:
                return "Text/String"
            elif 'int' in dtype_str:
                return "Integer"
            elif 'float' in dtype_str:
                return "Decimal"
            elif 'bool' in dtype_str:
                return "Boolean"
            elif 'datetime' in dtype_str:
                return "Date/Time"
            else:
                return "Other"
        except:
            return "Other"
    
    def calculate_quality_scores(col_data):
        """Calculate quality scores for a column"""
        total_count = len(col_data)
        null_count = col_data.isnull().sum()
        distinct_count = col_data.nunique()
        
        # Completeness Score
        completeness = ((total_count - null_count) / total_count) * 100 if total_count > 0 else 0
        
        # Uniqueness Score
        uniqueness = (distinct_count / total_count) * 100 if total_count > 0 else 0
        
        # Consistency Score
        consistency = 100.0
        non_null_data = col_data.dropna()
        
        if len(non_null_data) == 0:
            consistency = 0.0
        elif pd.api.types.is_object_dtype(col_data):
            # For text data, check for common consistency issues
            sample = non_null_data.head(200)
            issues_count = 0
            
            for value in sample:
                try:
                    str_value = str(value)
                    # Check for mixed case, extra spaces, special characters
                    if str_value != str_value.strip():
                        issues_count += 1
                    elif str_value.lower() != str_value and str_value.upper() != str_value:
                        # Mixed case (not all lower or all upper)
                        continue  # This is usually normal
                except:
                    issues_count += 1
            
            if len(sample) > 0:
                consistency = max(0, 100 - (issues_count / len(sample)) * 100)
        
        return {
            'completeness': round(completeness, 1),
            'uniqueness': round(uniqueness, 1),
            'consistency': round(consistency, 1)
        }
    
    # Calculate metrics for all columns
    all_column_details = {}
    for column in df.columns:
        col_data = df[column]
        data_type = classify_data_type(col_data)
        quality_scores = calculate_quality_scores(col_data)
        
        all_column_details[column] = {
            'data_type': data_type,
            'completeness': quality_scores['completeness'],
            'uniqueness': quality_scores['uniqueness'],
            'consistency': quality_scores['consistency']
        }
    
    # Calculate overall averages
    completeness_scores = [info['completeness'] for info in all_column_details.values()]
    uniqueness_scores = [info['uniqueness'] for info in all_column_details.values()]
    consistency_scores = [info['consistency'] for info in all_column_details.values()]
    
    overall_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
    overall_uniqueness = sum(uniqueness_scores) / len(uniqueness_scores) if uniqueness_scores else 0
    overall_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
    
    # Calculate column type distribution
    type_distribution = {}
    for column_info in all_column_details.values():
        data_type = column_info['data_type']
        type_distribution[data_type] = type_distribution.get(data_type, 0) + 1
    
    return {
        'completeness': round(overall_completeness, 1),
        'uniqueness': round(overall_uniqueness, 1),
        'consistency': round(overall_consistency, 1),
        'combined_score': round((overall_completeness + overall_uniqueness + overall_consistency) / 3, 1),
        'type_distribution': type_distribution
    }

def main():
    
    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'validation_config' not in st.session_state:
        st.session_state.validation_config = {}
    if 'critical_columns' not in st.session_state:
        st.session_state.critical_columns = []
    if 'report_generated' not in st.session_state:
        st.session_state.report_generated = False
    if 'checker' not in st.session_state:
        st.session_state.checker = None
    if 'historical_data' not in st.session_state:
        st.session_state.historical_data = None
    if 'tracking_enabled' not in st.session_state:
        st.session_state.tracking_enabled = False
    if 'dataset_snapshots' not in st.session_state:
        st.session_state.dataset_snapshots = {}

    # Main header banner
    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Quality Toolkit</h1>
        <p>Comprehensive data validation and quality reporting for your CSV datasets</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Download guide button below banner on bottom left
    col_guide, col_spacer = st.columns([1, 4])
    
    with col_guide:
        landing_page_path = "data_quality_service_landing.html"
        try:
            with open(landing_page_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.download_button(
                label="📥 Download Guide",
                data=html_content,
                file_name="data_quality_toolkit_guide.html",
                mime="text/html",
                help="Download comprehensive toolkit guide as HTML file"
            )
        except FileNotFoundError:
            pass  # Silently skip if file doesn't exist

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Upload Data", "✅ Configure Validations", "⭐ Critical Columns", "📊 Generate Report", "📈 Historical Tracking"])

    # Tab 1: Upload Data
    with tab1:
        st.header("📁 Upload Your CSV File")
        
        # Tips section at the top
        st.markdown("""
        <div class="feature-card">
            <h3>💡 Tips for Best Results</h3>
            <ul>
                <li>Ensure first row contains column headers</li>
                <li>Use consistent date formats</li>
                <li>Avoid special characters in column names</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=['csv'],
            help="Upload your CSV file to begin data quality analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                
                st.markdown('<div class="success-message">✅ File uploaded successfully!</div>', unsafe_allow_html=True)
                
                # Display basic info
                st.subheader("📋 Dataset Overview")
                
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.markdown('<div class="metric-card"><h3>{}</h3><p>Rows</p></div>'.format(len(df)), unsafe_allow_html=True)
                with col_b:
                    st.markdown('<div class="metric-card"><h3>{}</h3><p>Columns</p></div>'.format(len(df.columns)), unsafe_allow_html=True)
                with col_c:
                    st.markdown('<div class="metric-card"><h3>{}</h3><p>Data Types</p></div>'.format(df.dtypes.nunique()), unsafe_allow_html=True)
                with col_d:
                    st.markdown('<div class="metric-card"><h3>{:.1f}%</h3><p>Complete</p></div>'.format((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100), unsafe_allow_html=True)
                
                # Show data preview
                st.subheader("👀 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Show column information
                st.subheader("📊 Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Unique Values': df.nunique()
                })
                st.dataframe(col_info, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                st.info("Please ensure your file is a valid CSV format.")

    # Tab 2: Configure Validations
    with tab2:
        st.header("✅ Configure Validation Rules (Optional)")
        
        if st.session_state.df is None:
            st.warning("Please upload a CSV file first in the 'Upload Data' tab.")
        else:
            df = st.session_state.df
            
            st.markdown("""
            <div class="step-container">
                <h3>🎯 Validation Configuration</h3>
                <p>Configure custom validation rules for your data. If you skip this step, standard validations will be automatically applied.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize validation variables
            not_null_columns = []
            unique_columns = []
            range_validations = {}
            allowed_values_validations = {}
            regex_validations = {}
            
            # Section 1: Basic Validations
            with st.expander("🔍 **Basic Data Validations**", expanded=True):
                st.markdown("*Set up fundamental data quality checks*")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### 🚫 Not Null Checks")
                    st.markdown("*Ensure critical fields are never empty*")
                    not_null_columns = st.multiselect(
                        "Select columns that should never be empty:",
                        options=df.columns.tolist(),
                        key="not_null_cols",
                        help="These columns will be flagged if they contain missing values"
                    )
                
                with col2:
                    st.markdown("##### 🆔 Uniqueness Checks")
                    st.markdown("*Prevent duplicate values in key fields*")
                    unique_columns = st.multiselect(
                        "Select columns that should have unique values:",
                        options=df.columns.tolist(),
                        key="unique_cols",
                        help="These columns will be checked for duplicate entries"
                    )
            
            # Section 2: Numeric Validations  
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                with st.expander("📊 **Numeric Range Validations**", expanded=False):
                    st.markdown("*Set acceptable ranges for numeric columns*")
                    
                    if len(numeric_cols) > 0:
                        # Show numeric columns in a cleaner grid
                        num_cols_per_row = 2
                        for i in range(0, len(numeric_cols), num_cols_per_row):
                            cols = st.columns(num_cols_per_row)
                            for j, col_name in enumerate(numeric_cols[i:i+num_cols_per_row]):
                                with cols[j]:
                                    if st.checkbox(f"📊 Set range for **{col_name}**", key=f"range_{col_name}"):
                                        # Show current data stats
                                        col_min_val = df[col_name].min()
                                        col_max_val = df[col_name].max()
                                        st.caption(f"Current range: {col_min_val:.2f} to {col_max_val:.2f}")
                                        
                                        min_val = st.number_input(
                                            f"Minimum value:", 
                                            value=float(col_min_val),
                                            key=f"min_{col_name}"
                                        )
                                        max_val = st.number_input(
                                            f"Maximum value:", 
                                            value=float(col_max_val),
                                            key=f"max_{col_name}"
                                        )
                                        range_validations[col_name] = (min_val, max_val)
                    else:
                        st.info("No numeric columns found in your dataset.")
            
            # Section 3: Categorical Validations
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                with st.expander("📝 **Categorical Value Validations**", expanded=False):
                    st.markdown("*Define acceptable values for text/categorical columns*")
                    
                    for col in categorical_cols[:5]:  # Limit to first 5 for performance
                        if st.checkbox(f"📝 Set allowed values for **{col}**", key=f"allowed_{col}"):
                            unique_vals = df[col].dropna().unique()[:20]  # Show max 20 unique values
                            
                            # Show preview of current values
                            st.caption(f"Found {len(df[col].unique())} unique values. Showing first 20:")
                            
                            selected_vals = st.multiselect(
                                f"Select allowed values for {col}:",
                                options=unique_vals.tolist(),
                                default=unique_vals.tolist(),
                                key=f"vals_{col}",
                                help=f"Values not in this list will be flagged as invalid"
                            )
                            if selected_vals:
                                allowed_values_validations[col] = selected_vals
                    
                    if len(categorical_cols) > 5:
                        st.info(f"Showing first 5 categorical columns. Your dataset has {len(categorical_cols)} text columns total.")
            
            # Section 4: Pattern Validations
            if len(categorical_cols) > 0:
                with st.expander("🔍 **Pattern & Format Validations**", expanded=False):
                    st.markdown("*Validate text formats using patterns (email, phone, etc.)*")
                    
                    # Enhanced pattern presets with more options
                    pattern_presets = {
                        "Email Address": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                        "US Phone Number": r'^\(\d{3}\)\s\d{3}-\d{4}$',
                        "International Phone": r'^\+?[\d\s\-\(\)]{7,15}$',
                        "ZIP Code (US)": r'^\d{5}(-\d{4})?$',
                        "Postal Code (Canada)": r'^[A-Za-z]\d[A-Za-z][ -]?\d[A-Za-z]\d$',
                        "Social Security Number": r'^\d{3}-\d{2}-\d{4}$',
                        "Credit Card Number": r'^\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}$',
                        "URL/Website": r'^https?://(?:[-\w.])+(?:\.[a-zA-Z]{2,})+(?:/[^?\s]*)?(?:\?[^#\s]*)?(?:#[^\s]*)?$',
                        "IP Address": r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
                        "Date (YYYY-MM-DD)": r'^\d{4}-\d{2}-\d{2}$',
                        "Date (MM/DD/YYYY)": r'^\d{2}/\d{2}/\d{4}$',
                        "Time (HH:MM)": r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$',
                        "Currency (USD)": r'^\$?[\d,]+\.?\d{0,2}$',
                        "Alphanumeric Code": r'^[A-Za-z0-9]+$',
                        "Letters Only": r'^[A-Za-z\s]+$',
                        "Numbers Only": r'^\d+$',
                        "Custom Pattern": ""
                    }
                    
                    # Column selection for pattern validation
                    if len(categorical_cols) > 8:
                        st.info(f"Your dataset has {len(categorical_cols)} text columns. Use the multiselect below to choose which columns to configure.")
                        selected_pattern_cols = st.multiselect(
                            "Select columns for pattern validation:",
                            options=categorical_cols,
                            key="pattern_columns_select",
                            help="Choose which text columns you want to apply pattern validation to"
                        )
                    else:
                        st.info(f"Found {len(categorical_cols)} text columns in your dataset.")
                        selected_pattern_cols = categorical_cols
                    
                    # Pattern configuration for selected columns
                    for col in selected_pattern_cols:
                        if st.checkbox(f"🔍 Set pattern for **{col}**", key=f"pattern_{col}"):
                            
                            col_pattern, col_preview = st.columns([2, 1])
                            
                            with col_pattern:
                                pattern_type = st.selectbox(
                                    f"Choose pattern type for {col}:",
                                    options=list(pattern_presets.keys()),
                                    key=f"pattern_type_{col}",
                                    help="Select a common pattern or create a custom one"
                                )
                                
                                if pattern_type == "Custom Pattern":
                                    pattern = st.text_input(
                                        f"Enter regex pattern for {col}:", 
                                        key=f"custom_pattern_{col}",
                                        help="Use regex syntax (e.g., ^[A-Z]{2}\\d{4}$ for state codes like CA1234)",
                                        placeholder="Enter your custom regex pattern..."
                                    )
                                else:
                                    pattern = pattern_presets[pattern_type]
                                    st.code(pattern, language="regex")
                                    st.caption(f"Selected pattern: **{pattern_type}**")
                            
                            with col_preview:
                                if pattern:
                                    # Show sample values from the column
                                    sample_vals = df[col].dropna().head(5).tolist()
                                    st.caption("**Sample values:**")
                                    for i, val in enumerate(sample_vals):
                                        # Check if sample matches pattern
                                        try:
                                            matches = bool(re.match(pattern, str(val)))
                                            icon = "✅" if matches else "❌"
                                            st.caption(f"{icon} {val}")
                                        except re.error:
                                            st.caption(f"• {val}")
                                    
                                    # Show stats
                                    if pattern and pattern.strip():
                                        try:
                                            total_non_null = df[col].dropna().shape[0]
                                            matching_count = df[col].dropna().astype(str).str.match(pattern).sum()
                                            match_percentage = (matching_count / total_non_null * 100) if total_non_null > 0 else 0
                                            st.caption(f"**Match rate:** {matching_count}/{total_non_null} ({match_percentage:.1f}%)")
                                        except re.error:
                                            st.caption("Invalid regex pattern")
                                
                                if pattern and pattern.strip():
                                    regex_validations[col] = pattern
        
            # Save configuration
            if st.button("💾 Save Validation Configuration", type="primary"):
                config = {
                    "not_null": not_null_columns,
                    "unique": unique_columns,
                    "ranges": range_validations,
                    "allowed_values": allowed_values_validations,
                    "patterns": regex_validations
                }
                st.session_state.validation_config = config
                st.success("✅ Validation configuration saved!")
                
                # Show summary
                with st.expander("📋 Configuration Summary", expanded=True):
                    st.json(config)

    # Tab 3: Critical Columns
    with tab3:
        st.header("⭐ Mark Critical Data Elements (Optional)")
        
        if st.session_state.df is None:
            st.warning("Please upload a CSV file first in the 'Upload Data' tab.")
        else:
            df = st.session_state.df
            
            st.markdown("""
            <div class="step-container">
                <h3>🎯 Critical Column Selection</h3>
                <p>Our algorithm has analyzed your data to suggest critical columns based on data quality metrics (completeness ≥80%). These suggestions match the same logic used in the generated reports.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Function to detect critical columns using the SAME algorithm as the data quality checker
            def suggest_critical_columns(df):
                """Suggest critical columns based on the same algorithm used in the data quality checker"""
                suggestions = []
                analysis_details = {}
                
                for col in df.columns:
                    col_data = df[col]
                    total_count = len(col_data)
                    null_count = col_data.isnull().sum()
                    
                    # Calculate completeness percentage (same as data quality checker)
                    completeness = ((total_count - null_count) / total_count) * 100 if total_count > 0 else 0
                    
                    # Apply the EXACT same algorithm from the data quality checker:
                    # A column is considered critical if:
                    # 1. It has high completeness (>= 80%)
                    # 2. It's not mostly empty (null_count < total_count * 0.5)
                    is_critical_by_algorithm = (
                        completeness >= 80 and 
                        null_count < total_count * 0.5
                    )
                    
                    # Determine confidence level and reasons
                    reasons = []
                    confidence_score = 0
                    
                    if is_critical_by_algorithm:
                        confidence_score = 3  # Base score for meeting algorithm criteria
                        reasons.append("Meets quality threshold")
                        
                        # Additional scoring for confidence levels
                        if completeness >= 95:
                            confidence_score += 2
                            reasons.append("Excellent completeness")
                        elif completeness >= 90:
                            confidence_score += 1
                            reasons.append("High completeness")
                        
                        if completeness >= 99:
                            confidence_score += 1
                            reasons.append("Near-perfect completeness")
                    else:
                        # Explain why it didn't meet criteria
                        if completeness < 80:
                            reasons.append(f"Low completeness ({completeness:.1f}%)")
                        if null_count >= total_count * 0.5:
                            reasons.append("Too many null values")
                    
                    analysis_details[col] = {
                        'is_critical': is_critical_by_algorithm,
                        'score': confidence_score,
                        'reasons': reasons,
                        'completeness': completeness,
                        'null_count': null_count,
                        'total_count': total_count
                    }
                    
                    if is_critical_by_algorithm:
                        suggestions.append(col)
                
                # Sort suggestions by confidence score (higher scores first)
                suggestions.sort(key=lambda col: analysis_details[col]['score'], reverse=True)
                
                return suggestions, analysis_details
            
            # Get suggestions
            suggested_columns, analysis_scores = suggest_critical_columns(df)
            
            # Algorithmic suggestions section
            with st.expander("⚙️ **Algorithm Suggestions - Recommended Critical Columns**", expanded=True):
                st.markdown("*Based on data quality metrics: completeness ≥80% and low null percentage*")
                
                # Color schema key
                st.markdown("""
                <div style="background: rgba(255, 255, 255, 0.03); padding: 0.75rem; border-radius: 6px; margin: 1rem 0;">
                    <small><strong>Color Key:</strong> 
                    <span style="color: #28a745;">• Green</span>: ≥95% completeness | 
                    <span style="color: #ffc107;">• Yellow</span>: ≥90% completeness | 
                    <span style="color: #fd7e14;">• Orange</span>: ≥80% completeness
                    </small>
                </div>
                """, unsafe_allow_html=True)
                
                if suggested_columns:
                    st.markdown("**We recommend these columns as critical:**")
                    
                    # Show suggested columns with analysis
                    suggestion_cols = st.columns(min(len(suggested_columns), 3))
                    for i, col in enumerate(suggested_columns):
                        with suggestion_cols[i % 3]:
                            analysis = analysis_scores[col]
                            
                            # Color coding based on completeness - matches color key
                            completeness = analysis['completeness']
                            if completeness >= 95:
                                confidence = "🟢 High Confidence"
                                bg_color = "rgba(34, 197, 94, 0.15)"
                                border_color = "#28a745"
                                text_color = "#4ade80"
                            elif completeness >= 90:
                                confidence = "🟡 Medium Confidence" 
                                bg_color = "rgba(255, 193, 7, 0.15)"
                                border_color = "#ffc107"
                                text_color = "#ffc107"
                            else:
                                confidence = "🟠 Low Confidence"
                                bg_color = "rgba(253, 126, 20, 0.15)"
                                border_color = "#fd7e14"
                                text_color = "#fd7e14"
                            
                            st.markdown(f"""
                            <div style="background: {bg_color}; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border: 1px solid {border_color}; backdrop-filter: blur(10px);">
                                <strong style="color: #e0e6ed;">📌 {col}</strong><br>
                                <small style="color: {text_color}; font-weight: 600;">{confidence}</small><br>
                                <small style="color: #e0e6ed;">• {', '.join(analysis['reasons'][:2])}</small><br>
                                <small style="color: #9ca3af;">• {analysis['completeness']:.1f}% complete</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                else:
                    st.info("No columns meet the algorithm criteria (≥80% completeness). You can still select critical columns manually below.")
            
            # Manual selection section
            with st.expander("⚙️ **Manual Selection - Fine-tune Your Critical Columns**", expanded=True):
                st.markdown("*Review and modify the critical column selection*")
                
                # Pre-populate with suggestions if no manual selection yet
                default_selection = st.session_state.critical_columns if st.session_state.critical_columns else suggested_columns
                
                critical_columns = st.multiselect(
                    "Choose columns that are critical for your business:",
                    options=df.columns.tolist(),
                    default=default_selection,
                    help="Critical columns get priority analysis and enhanced monitoring",
                    key="manual_critical_selection"
                )
                
                # Show selected columns in a grid
                if critical_columns:
                    st.markdown("**📊 Selected Critical Columns:**")
                    
                    # Create grid layout - 4 columns per row
                    cols_per_row = 4
                    for i in range(0, len(critical_columns), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col_name in enumerate(critical_columns[i:i+cols_per_row]):
                            with cols[j]:
                                analysis = analysis_scores.get(col_name, {})
                                null_count = df[col_name].isnull().sum()
                                completeness = ((len(df) - null_count) / len(df)) * 100
                                
                                # Determine recommendation status
                                if col_name in suggested_columns:
                                    border_color = "#28a745"
                                    status_icon = "✅"
                                    status_text = "Recommended"
                                else:
                                    border_color = "#6c757d"
                                    status_icon = "ℹ️"
                                    status_text = "Manual"
                                
                                # Create square card
                                st.markdown(f"""
                                <div style="
                                    background: rgba(255, 255, 255, 0.05);
                                    border: 2px solid {border_color};
                                    border-radius: 8px;
                                    padding: 1rem;
                                    text-align: center;
                                    margin: 0.5rem 0;
                                    min-height: 120px;
                                    display: flex;
                                    flex-direction: column;
                                    justify-content: center;
                                ">
                                    <div style="font-size: 1.2rem; font-weight: bold; color: #e5e7eb; margin-bottom: 0.5rem;">
                                        {status_icon} {col_name}
                                    </div>
                                    <div style="font-size: 0.8rem; color: #9ca3af;">
                                        {status_text}<br>
                                        {completeness:.1f}% complete
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Save button
                if st.button("💾 Save Critical Columns", type="primary"):
                    st.session_state.critical_columns = critical_columns
                    st.success(f"✅ Saved {len(critical_columns)} critical columns!")
                    

    # Tab 4: Generate Report
    with tab4:
        st.header("📊 Generate Data Quality Report")
        
        if st.session_state.df is None:
            st.warning("Please upload a CSV file first in the 'Upload Data' tab.")
        else:
            df = st.session_state.df
            
            st.markdown("""
            <div class="step-container">
                <h3>🚀 Ready to Generate Report</h3>
                <p>Your data is ready for analysis. Click the button below to generate a comprehensive data quality report.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Report options
            st.subheader("📋 Report Configuration")
            
            report_title = st.text_input(
                "Report Title:",
                value=f"Data Quality Report - {datetime.now().strftime('%Y-%m-%d')}",
                help="This will appear as the main title in your report"
            )
            
            dataset_name = st.text_input(
                "Dataset Name:",
                value="Uploaded Dataset",
                help="A descriptive name for your dataset"
            )
            
            # Generate report button
            if st.button("🚀 Generate Data Quality Report", type="primary", width="stretch"):
                # Clear any existing success message
                st.session_state.show_success_message = False
                
                with st.spinner("🔄 Analyzing your data and generating report..."):
                    try:
                        # Create DataQualityChecker instance
                        checker = DataQualityChecker(
                            df=df, 
                            dataset_name=dataset_name,
                            critical_columns=st.session_state.critical_columns
                        )
                        
                        # Apply validations based on configuration
                        config = st.session_state.validation_config
                        
                        # Apply not null validations
                        for col in config.get('not_null', []):
                            if col in df.columns:
                                checker.expect_column_values_to_not_be_null(col)
                        
                        # Apply uniqueness validations
                        for col in config.get('unique', []):
                            if col in df.columns:
                                checker.expect_column_values_to_be_unique(col)
                        
                        # Apply range validations
                        for col, (min_val, max_val) in config.get('ranges', {}).items():
                            if col in df.columns:
                                checker.expect_column_values_to_be_in_range(col, min_val, max_val)
                        
                        # Apply allowed values validations
                        for col, allowed_vals in config.get('allowed_values', {}).items():
                            if col in df.columns and allowed_vals:
                                checker.expect_column_values_to_be_in_set(col, allowed_vals)
                        
                        # Apply regex pattern validations
                        for col, pattern in config.get('patterns', {}).items():
                            if col in df.columns and pattern:
                                checker.expect_column_values_to_match_regex(col, pattern)
                        
                        # Generate HTML report
                        report_html = checker.generate_data_docs(
                            title=report_title,
                            dataset_name=dataset_name
                        )
                        
                        st.session_state.report_generated = True
                        st.session_state.report_html = report_html
                        st.session_state.report_title = report_title
                        st.session_state.checker = checker
                        
                        # Add to historical tracking if enabled
                        if st.session_state.tracking_enabled:
                            # Get version and options from the historical tracking tab
                            dataset_version = st.session_state.get('dataset_version', 'v1.0')
                            enable_data_export = st.session_state.get('enable_data_export', True)
                            enable_detailed_comparison = st.session_state.get('enable_detailed_comparison', False)
                            
                            add_tracking_entry(checker, dataset_name, df, dataset_version, enable_data_export, enable_detailed_comparison)
                        
                        # Set success message with timestamp
                        st.session_state.success_message_time = datetime.now()
                        st.session_state.show_success_message = True
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
            
            # Show success message with auto-hide
            if st.session_state.get('show_success_message', False):
                success_time = st.session_state.get('success_message_time')
                if success_time:
                    time_diff = (datetime.now() - success_time).total_seconds()
                    if time_diff < 3:  # Show for 3 seconds
                        st.markdown("""
                        <style>
                        .auto-fade-success {
                            background: rgba(34, 197, 94, 0.1);
                            color: #4ade80;
                            padding: 1rem;
                            border-radius: 5px;
                            border-left: 5px solid #22c55e;
                            margin: 1rem 0;
                            border: 1px solid rgba(34, 197, 94, 0.2);
                            animation: fadeOutCollapse 3s ease-in-out forwards;
                            overflow: hidden;
                        }
                        
                        @keyframes fadeOutCollapse {
                            0% { 
                                opacity: 1; 
                                max-height: 100px; 
                                margin: 1rem 0; 
                                padding: 1rem; 
                            }
                            70% { 
                                opacity: 1; 
                                max-height: 100px; 
                                margin: 1rem 0; 
                                padding: 1rem; 
                            }
                            90% { 
                                opacity: 0; 
                                max-height: 100px; 
                                margin: 1rem 0; 
                                padding: 1rem; 
                            }
                            100% { 
                                opacity: 0; 
                                max-height: 0; 
                                margin: 0; 
                                padding: 0; 
                                visibility: hidden; 
                            }
                        }
                        </style>
                        <div class="auto-fade-success">
                            ✅ Report generated successfully!
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Hide the message after 3 seconds
                        st.session_state.show_success_message = False
            
            # Display generated report
            if st.session_state.report_generated:
                st.markdown("---")
                # Use the custom report title from session state
                report_title = st.session_state.get('report_title', 'Your Data Quality Report')
                st.subheader(f"📊 {report_title}")
                
                # Download buttons - all side by side
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    # Download HTML report
                    html_bytes = st.session_state.report_html.encode('utf-8')
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=html_bytes,
                        file_name=f"{st.session_state.report_title.replace(' ', '_')}.html",
                        mime="text/html",
                        type="primary",
                        help="Complete interactive report with charts, visualizations, and detailed analysis",
                        use_container_width=True
                    )
                
                with col_b:
                    # Download validation results as CSV
                    if st.session_state.checker is not None:
                        results_df = st.session_state.checker.get_results()
                        if not results_df.empty:
                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="📊 Download Validation Results",
                                data=csv_buffer.getvalue(),
                                file_name=f"validation_results_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                help="Summary of validation rules with success rates and pass/fail statistics",
                                use_container_width=True
                            )
                
                with col_c:
                    # Download enhanced dataset with validation columns
                    if st.session_state.checker is not None and hasattr(st.session_state, 'df'):
                        enhanced_df = create_enhanced_dataset(st.session_state.df, st.session_state.checker, st.session_state.validation_config)
                        csv_buffer = io.StringIO()
                        enhanced_df.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📈 Download Enhanced Dataset",
                            data=csv_buffer.getvalue(),
                            file_name=f"enhanced_dataset_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            help="Original data with validation pass/fail columns appended",
                            use_container_width=True
                        )
                
                # Show results summary
                if st.session_state.checker is not None:
                    results_df = st.session_state.checker.get_results()
                    df = st.session_state.df
                    
                    # Calculate overall data quality metrics
                    overall_metrics = calculate_overall_data_quality(df)
                    
                    st.subheader("📈 Results Summary")
                    
                    # Section 1: Data Quality Metrics (Expandable)
                    with st.expander("📊 **Data Quality Metrics**", expanded=True):
                        col_table, col_score = st.columns([3, 1])
                        
                        with col_table:
                            # Clean metrics table (removed consistency)
                            quality_data = {
                                'Metric': ['Completeness', 'Uniqueness'],
                                'Score': [f"{overall_metrics['completeness']:.1f}%", 
                                         f"{overall_metrics['uniqueness']:.1f}%"]
                            }
                            quality_df = pd.DataFrame(quality_data)
                            st.dataframe(
                                quality_df,
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        with col_score:
                            # Square overall score box
                            st.markdown(f"""
                            <div style="
                                background: rgba(255, 255, 255, 0.05);
                                border: 1px solid rgba(255, 255, 255, 0.1);
                                border-radius: 8px;
                                padding: 1.5rem;
                                text-align: center;
                                height: 140px;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;
                                align-items: center;
                            ">
                                <div style="
                                    color: #e5e7eb;
                                    font-size: 2.5rem;
                                    font-weight: bold;
                                    margin-bottom: 0.5rem;
                                ">{round((overall_metrics['completeness'] + overall_metrics['uniqueness']) / 2, 1):.1f}%</div>
                                <div style="
                                    color: #9ca3af;
                                    font-size: 1rem;
                                ">Overall Score</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Section 2: Data Types (Expandable)
                    with st.expander("🔢 **Data Type Distribution**", expanded=False):
                        type_dist = overall_metrics['type_distribution']
                        
                        if type_dist:
                            # Create a simple table for data types
                            type_data = {
                                'Data Type': list(type_dist.keys()),
                                'Column Count': list(type_dist.values()),
                                'Percentage': [f"{(count / len(df.columns)) * 100:.1f}%" 
                                              for count in type_dist.values()]
                            }
                            type_df = pd.DataFrame(type_data)
                            st.dataframe(
                                type_df,
                                use_container_width=True,
                                hide_index=True
                            )
                    
                    # Section 3: Validation Results (Expandable)
                    if not results_df.empty:
                        with st.expander("✅ **Validation Performance**", expanded=False):
                            # Simple validation metrics
                            total_validations = len(results_df)
                            avg_success_rate = results_df['success_rate'].mean()
                            passed_validations = (results_df['success_rate'] >= 90).sum()
                            
                            col_x, col_y, col_z = st.columns(3)
                            with col_x:
                                st.metric("Total Validations", total_validations)
                            with col_y:
                                st.metric("Average Success Rate", f"{avg_success_rate:.1f}%")
                            with col_z:
                                st.metric("Passed Validations", f"{passed_validations}/{total_validations}")
                            
                            # Results table
                            st.markdown("**Validation Details:**")
                            st.dataframe(
                                results_df[['column', 'rule', 'success_rate']].round(1),
                                use_container_width=True
                            )
                
                # Preview report
                st.subheader("👀 Report Preview")
                st.info("💡 The full interactive report is available in the downloaded HTML file. Below is a simplified preview.")
                
                # Display the HTML report in an iframe-like container
                with st.container():
                    st.components.v1.html(
                        st.session_state.report_html,
                        height=800,
                        scrolling=True
                    )

    # Tab 5: Historical Tracking - Snapshot-Based Analysis
    with tab5:
        st.header("📈 Historical Data Quality Analysis")
        
        # Introduction
        st.markdown("""
        <div class="step-container">
            <h3>📊 Snapshot-Based Quality Tracking</h3>
            <p>Upload multiple versions or snapshots of the same dataset to analyze data quality trends over time. Compare completeness, uniqueness, consistency, and overall scores across different versions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for snapshots
        if 'dataset_snapshots' not in st.session_state:
            st.session_state.dataset_snapshots = []
        
        # Snapshot upload section
        st.subheader("📁 Upload Dataset Snapshots")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File uploader for multiple snapshots
            uploaded_snapshots = st.file_uploader(
                "Upload Multiple Dataset Snapshots (CSV files)",
                type=['csv'],
                accept_multiple_files=True,
                help="Upload 2 or more snapshots of the same dataset to perform historical analysis",
                key="snapshot_upload"
            )
            
            # Process uploaded snapshots
            if uploaded_snapshots and len(uploaded_snapshots) >= 2:
                if st.button("🔄 Process Snapshots", help="Analyze uploaded snapshots for historical comparison"):
                    snapshots_data = []
                    
                    with st.spinner("Processing snapshots..."):
                        for i, uploaded_file in enumerate(uploaded_snapshots):
                            try:
                                # Read the CSV file
                                df = pd.read_csv(uploaded_file)
                                
                                # Calculate data quality metrics using the existing function
                                quality_metrics = calculate_overall_data_quality(df)
                                
                                # Determine snapshot identifier from filename or use index
                                snapshot_name = uploaded_file.name.replace('.csv', '')
                                
                                # Create snapshot entry (removed consistency as it's not meaningful)
                                snapshot_data = {
                                    'snapshot_id': f"snapshot_{i+1}",
                                    'name': snapshot_name,
                                    'data': df,
                                    'timestamp': datetime.now() - pd.Timedelta(days=len(uploaded_snapshots)-i-1),  # Simulate time progression
                                    'row_count': len(df),
                                    'column_count': len(df.columns),
                                    'completeness': quality_metrics['completeness'],
                                    'uniqueness': quality_metrics['uniqueness'],
                                    'overall_score': round((quality_metrics['completeness'] + quality_metrics['uniqueness']) / 2, 1),  # Updated calculation without consistency
                                    'type_distribution': quality_metrics['type_distribution']
                                }
                                
                                snapshots_data.append(snapshot_data)
                                
                            except Exception as e:
                                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    
                    # Store snapshots in session state
                    st.session_state.dataset_snapshots = snapshots_data
                    st.success(f"✅ Successfully processed {len(snapshots_data)} snapshots!")
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>📋 Requirements</h3>
                <ul>
                    <li><strong>2+ snapshots</strong> - Minimum for comparison</li>
                    <li><strong>Same schema</strong> - Similar column structure</li>
                    <li><strong>CSV format</strong> - Standard data format</li>
                    <li><strong>Sequential naming</strong> - For time ordering</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Display analysis if snapshots are available
        if st.session_state.dataset_snapshots and len(st.session_state.dataset_snapshots) >= 2:
            st.markdown("---")
            st.subheader("📊 Historical Quality Analysis")
            
            snapshots = st.session_state.dataset_snapshots
            
            # Create comparison dataframe (removed consistency)
            comparison_data = []
            for snapshot in snapshots:
                comparison_data.append({
                    'Snapshot': snapshot['name'],
                    'Timestamp': snapshot['timestamp'].strftime('%Y-%m-%d %H:%M'),
                    'Rows': snapshot['row_count'],
                    'Columns': snapshot['column_count'],
                    'Completeness %': snapshot['completeness'],
                    'Uniqueness %': snapshot['uniqueness'],
                    'Overall Score %': snapshot['overall_score']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Overview metrics
            with st.expander("📋 Snapshot Overview", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Snapshots", len(snapshots))
                
                with col2:
                    avg_score = comparison_df['Overall Score %'].mean()
                    score_trend = "↗️" if comparison_df['Overall Score %'].iloc[-1] > comparison_df['Overall Score %'].iloc[0] else "↘️"
                    st.metric("Avg Quality Score", f"{avg_score:.1f}%", f"{score_trend}")
                
                with col3:
                    row_change = comparison_df['Rows'].iloc[-1] - comparison_df['Rows'].iloc[0]
                    st.metric("Data Volume Change", f"{comparison_df['Rows'].iloc[-1]:,}", f"{row_change:+,}")
                
                with col4:
                    col_change = comparison_df['Columns'].iloc[-1] - comparison_df['Columns'].iloc[0]
                    st.metric("Schema Change", comparison_df['Columns'].iloc[-1], f"{col_change:+}")
                
                # Summary table
                st.dataframe(comparison_df, use_container_width=True)
            
            # Quality trends charts
            with st.expander("📈 Quality Trends", expanded=False):
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.markdown("**Overall Quality Score Trend**")
                    chart_data = comparison_df.set_index('Snapshot')[['Overall Score %']]
                    st.line_chart(chart_data)
                
                with chart_col2:
                    st.markdown("**Individual Quality Metrics**")
                    metrics_chart = comparison_df.set_index('Snapshot')[['Completeness %', 'Uniqueness %']]
                    st.line_chart(metrics_chart)
                
                # Data volume trends
                st.markdown("**Data Volume & Schema Changes**")
                volume_chart = comparison_df.set_index('Snapshot')[['Rows', 'Columns']]
                st.line_chart(volume_chart)
            
            # Data type distribution analysis
            with st.expander("🗂️ Data Type Distribution Analysis", expanded=False):
                st.markdown("**Data Type Changes Across Snapshots**")
                
                type_analysis = []
                for snapshot in snapshots:
                    for data_type, count in snapshot['type_distribution'].items():
                        type_analysis.append({
                            'Snapshot': snapshot['name'],
                            'Data Type': data_type,
                            'Count': count,
                            'Percentage': round((count / snapshot['column_count']) * 100, 1)
                        })
                
                type_df = pd.DataFrame(type_analysis)
                
                if not type_df.empty:
                    # Pivot table for better visualization
                    type_pivot = type_df.pivot(index='Snapshot', columns='Data Type', values='Count').fillna(0)
                    st.dataframe(type_pivot, use_container_width=True)
                    
                    # Chart showing type distribution trends
                    st.bar_chart(type_pivot)
            
            # Detailed snapshot comparison
            with st.expander("🔍 Detailed Snapshot Comparison", expanded=False):
                if len(snapshots) >= 2:
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        snapshot_a_name = st.selectbox(
                            "Select First Snapshot:",
                            options=[s['name'] for s in snapshots],
                            key="compare_a"
                        )
                    
                    with comp_col2:
                        snapshot_b_name = st.selectbox(
                            "Select Second Snapshot:",
                            options=[s['name'] for s in snapshots],
                            index=min(1, len(snapshots)-1),
                            key="compare_b"
                        )
                    
                    # Find selected snapshots
                    snapshot_a = next((s for s in snapshots if s['name'] == snapshot_a_name), None)
                    snapshot_b = next((s for s in snapshots if s['name'] == snapshot_b_name), None)
                    
                    if snapshot_a and snapshot_b and snapshot_a != snapshot_b:
                        st.markdown(f"**Comparing: {snapshot_a_name} vs {snapshot_b_name}**")
                        
                        # Quality comparison metrics
                        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                        
                        with comp_col1:
                            quality_diff = snapshot_b['overall_score'] - snapshot_a['overall_score']
                            st.metric(
                                "Overall Score Change",
                                f"{snapshot_b['overall_score']:.1f}%",
                                f"{quality_diff:+.1f}%"
                            )
                        
                        with comp_col2:
                            completeness_diff = snapshot_b['completeness'] - snapshot_a['completeness']
                            st.metric(
                                "Completeness Change",
                                f"{snapshot_b['completeness']:.1f}%",
                                f"{completeness_diff:+.1f}%"
                            )
                        
                        with comp_col3:
                            uniqueness_diff = snapshot_b['uniqueness'] - snapshot_a['uniqueness']
                            st.metric(
                                "Uniqueness Change",
                                f"{snapshot_b['uniqueness']:.1f}%",
                                f"{uniqueness_diff:+.1f}%"
                            )
                        
                        with comp_col4:
                            # Show row count change instead of consistency
                            row_diff = snapshot_b['row_count'] - snapshot_a['row_count']
                            st.metric(
                                "Row Count Change",
                                f"{snapshot_b['row_count']:,}",
                                f"{row_diff:+,}"
                            )
                        
                        # Schema comparison
                        st.markdown("**Schema Comparison**")
                        
                        cols_a = set(snapshot_a['data'].columns)
                        cols_b = set(snapshot_b['data'].columns)
                        
                        added_cols = cols_b - cols_a
                        removed_cols = cols_a - cols_b
                        common_cols = cols_a & cols_b
                        
                        schema_col1, schema_col2, schema_col3 = st.columns(3)
                        
                        with schema_col1:
                            st.markdown(f"**Common Columns:** {len(common_cols)}")
                            if common_cols:
                                st.write(", ".join(sorted(list(common_cols))[:5]) + ("..." if len(common_cols) > 5 else ""))
                        
                        with schema_col2:
                            st.markdown(f"**Added Columns:** {len(added_cols)}")
                            if added_cols:
                                st.write(", ".join(sorted(list(added_cols))))
                        
                        with schema_col3:
                            st.markdown(f"**Removed Columns:** {len(removed_cols)}")
                            if removed_cols:
                                st.write(", ".join(sorted(list(removed_cols))))
            
            # Export functionality
            st.markdown("---")
            st.subheader("📥 Export Analysis")
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                # Download comparison data as CSV
                csv_buffer = io.StringIO()
                comparison_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📊 Download Quality Comparison CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"historical_quality_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    help="Download the complete historical quality analysis as CSV"
                )
            
            with export_col2:
                # Clear snapshots
                if st.button("🗺️ Clear All Snapshots", help="Remove all uploaded snapshots to start fresh"):
                    st.session_state.dataset_snapshots = []
                    st.success("✅ All snapshots cleared!")
                    st.rerun()
        
        # Instructions when no snapshots
        elif not st.session_state.dataset_snapshots:
            st.markdown("---")
            st.info("📁 Upload 2 or more dataset snapshots above to begin historical analysis")
            
            st.markdown("""
            <div class="feature-card">
                <h3>🚀 Getting Started</h3>
                <p><strong>Step 1:</strong> Collect multiple versions of your dataset (e.g., monthly exports, version updates)</p>
                <p><strong>Step 2:</strong> Save each version as a CSV file with descriptive names (e.g., data_jan2024.csv, data_feb2024.csv)</p>
                <p><strong>Step 3:</strong> Upload all files using the file uploader above</p>
                <p><strong>Step 4:</strong> Click "Process Snapshots" to generate your historical analysis</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Example benefits
            st.markdown("""
            <div class="feature-card">
                <h3>📈 Analysis Features</h3>
                <ul>
                    <li><strong>Quality Trends:</strong> Track completeness and uniqueness over time</li>
                    <li><strong>Schema Evolution:</strong> Monitor column additions, removals, and type changes</li>
                    <li><strong>Data Volume:</strong> Analyze dataset growth and size trends</li>
                    <li><strong>Comparative Metrics:</strong> Side-by-side snapshot comparisons</li>
                    <li><strong>Export Reports:</strong> Download analysis results as CSV</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def add_tracking_entry(checker, dataset_name, df, dataset_version='v1.0', enable_data_export=True, enable_detailed_comparison=False):
    """Add a new entry to the historical tracking data with versioning and data snapshots"""
    
    # Get validation results
    results_df = checker.get_results()
    
    # Calculate quality metrics
    total_validations = len(results_df) if not results_df.empty else 0
    passed_validations = (results_df['success_rate'] >= 90).sum() if not results_df.empty else 0
    overall_quality_score = results_df['success_rate'].mean() if not results_df.empty else 100.0
    
    # Calculate detailed field-level statistics if enabled
    field_statistics = {}
    schema_fingerprint = ""
    
    if enable_detailed_comparison:
        for col in df.columns:
            col_stats = {
                'data_type': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'unique_count': int(df[col].nunique()),
                'null_percentage': round((df[col].isnull().sum() / len(df)) * 100, 2)
            }
            
            # Add type-specific statistics
            if df[col].dtype in ['int64', 'float64']:
                col_stats.update({
                    'mean': round(df[col].mean(), 2) if not df[col].isnull().all() else None,
                    'median': round(df[col].median(), 2) if not df[col].isnull().all() else None,
                    'std': round(df[col].std(), 2) if not df[col].isnull().all() else None
                })
            
            field_statistics[col] = col_stats
        
        # Create schema fingerprint (column names + types)
        schema_fingerprint = "|".join([f"{col}:{df[col].dtype}" for col in sorted(df.columns)])
    
    # Store data snapshot if enabled
    snapshot_id = None
    if enable_data_export:
        snapshot_id = f"{dataset_name}_{dataset_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Store first 1000 rows for comparison (to manage memory)
        snapshot_data = df.head(1000).copy()
        st.session_state.dataset_snapshots[snapshot_id] = {
            'data': snapshot_data,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'version': dataset_version,
            'dataset_name': dataset_name,
            'full_row_count': len(df)
        }
        
        # Keep only last 10 snapshots to manage memory
        if len(st.session_state.dataset_snapshots) > 10:
            oldest_key = min(st.session_state.dataset_snapshots.keys())
            del st.session_state.dataset_snapshots[oldest_key]
    
    # Create new tracking entry
    new_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_name': dataset_name,
        'dataset_version': dataset_version,
        'overall_quality_score': round(overall_quality_score, 1),
        'total_validations': total_validations,
        'passed_validations': passed_validations,
        'row_count': len(df),
        'column_count': len(df.columns),
        'data_completeness': round((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100, 1),
        'schema_fingerprint': schema_fingerprint,
        'snapshot_id': snapshot_id,
        'has_detailed_stats': enable_detailed_comparison
    }
    
    # Add field statistics as JSON string if enabled
    if enable_detailed_comparison and field_statistics:
        new_entry['field_statistics'] = json.dumps(field_statistics)
    
    # Add to historical data
    if st.session_state.historical_data is None:
        # Create new historical dataframe
        st.session_state.historical_data = pd.DataFrame([new_entry])
    else:
        # Append to existing historical data
        new_row_df = pd.DataFrame([new_entry])
        st.session_state.historical_data = pd.concat([st.session_state.historical_data, new_row_df], ignore_index=True)
    
    # Keep only last 100 entries to manage memory
    if len(st.session_state.historical_data) > 100:
        st.session_state.historical_data = st.session_state.historical_data.tail(100).reset_index(drop=True)

def generate_historical_summary(historical_df):
    """Generate a simple HTML summary of historical data"""
    
    if historical_df.empty:
        return "<html><body><h1>No historical data available</h1></body></html>"
    
    # Calculate summary statistics
    total_runs = len(historical_df)
    avg_quality = historical_df.get('overall_quality_score', pd.Series([0])).mean()
    best_quality = historical_df.get('overall_quality_score', pd.Series([0])).max()
    worst_quality = historical_df.get('overall_quality_score', pd.Series([0])).min()
    
    # Create simple HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Historical Data Quality Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; color: #1e40af; border-bottom: 2px solid #1e40af; padding-bottom: 20px; }}
            .metric {{ display: inline-block; margin: 20px; padding: 20px; background: #f8f9fa; border-radius: 8px; text-align: center; }}
            .metric h3 {{ color: #1e40af; margin: 0; font-size: 2em; }}
            .metric p {{ color: #666; margin: 5px 0 0 0; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #1e40af; color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📈 Historical Data Quality Summary</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div style="text-align: center; margin: 30px 0;">
                <div class="metric">
                    <h3>{total_runs}</h3>
                    <p>Total Quality Checks</p>
                </div>
                <div class="metric">
                    <h3>{avg_quality:.1f}%</h3>
                    <p>Average Quality Score</p>
                </div>
                <div class="metric">
                    <h3>{best_quality:.1f}%</h3>
                    <p>Best Quality Score</p>
                </div>
                <div class="metric">
                    <h3>{worst_quality:.1f}%</h3>
                    <p>Worst Quality Score</p>
                </div>
            </div>
            
            <h2>📊 Recent Quality Checks</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Dataset</th>
                    <th>Quality Score</th>
                    <th>Validations</th>
                    <th>Data Rows</th>
                </tr>
    """
    
    # Add recent entries to table
    recent_entries = historical_df.tail(10)
    for _, row in recent_entries.iterrows():
        timestamp = pd.to_datetime(row.get('timestamp', '')).strftime('%Y-%m-%d %H:%M') if 'timestamp' in row else 'N/A'
        dataset = row.get('dataset_name', 'Unknown')
        quality = row.get('overall_quality_score', 0)
        validations = row.get('total_validations', 0)
        rows = row.get('row_count', 0)
        
        html_content += f"""
                <tr>
                    <td>{timestamp}</td>
                    <td>{dataset}</td>
                    <td>{quality:.1f}%</td>
                    <td>{validations}</td>
                    <td>{rows:,}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    </body>
    </html>
    """
    
    return html_content

def generate_version_comparison(historical_df, version_a, version_b):
    """Generate a detailed comparison report between two dataset versions"""
    
    # Filter data for each version (get most recent entry for each version)
    data_a = historical_df[historical_df['dataset_version'] == version_a].iloc[-1] if version_a in historical_df['dataset_version'].values else None
    data_b = historical_df[historical_df['dataset_version'] == version_b].iloc[-1] if version_b in historical_df['dataset_version'].values else None
    
    if data_a is None or data_b is None:
        return "<html><body><h1>Version comparison data not available</h1></body></html>"
    
    # Calculate differences
    quality_diff = data_b['overall_quality_score'] - data_a['overall_quality_score']
    row_diff = data_b['row_count'] - data_a['row_count']
    col_diff = data_b['column_count'] - data_a['column_count']
    completeness_diff = data_b['data_completeness'] - data_a['data_completeness']
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dataset Version Comparison: {version_a} vs {version_b}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; color: #1e40af; border-bottom: 2px solid #1e40af; padding-bottom: 20px; }}
            .comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 30px 0; }}
            .version-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border: 1px solid #dee2e6; }}
            .version-card h3 {{ color: #1e40af; margin-top: 0; }}
            .metric {{ margin: 10px 0; }}
            .metric strong {{ color: #495057; }}
            .diff-positive {{ color: #28a745; font-weight: bold; }}
            .diff-negative {{ color: #dc3545; font-weight: bold; }}
            .diff-neutral {{ color: #6c757d; font-weight: bold; }}
            .summary-box {{ background: #e7f3ff; padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 5px solid #1e40af; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #1e40af; color: white; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Dataset Version Comparison</h1>
                <h2>{version_a} vs {version_b}</h2>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary-box">
                <h3>🎯 Key Changes Summary</h3>
                <ul>
                    <li><strong>Quality Score:</strong> {data_a['overall_quality_score']:.1f}% → {data_b['overall_quality_score']:.1f}% 
                        <span class="{'diff-positive' if quality_diff > 0 else 'diff-negative' if quality_diff < 0 else 'diff-neutral'}">
                            ({quality_diff:+.1f}%)
                        </span>
                    </li>
                    <li><strong>Data Volume:</strong> {data_a['row_count']:,} → {data_b['row_count']:,} rows 
                        <span class="{'diff-positive' if row_diff > 0 else 'diff-negative' if row_diff < 0 else 'diff-neutral'}">
                            ({row_diff:+,})
                        </span>
                    </li>
                    <li><strong>Schema:</strong> {data_a['column_count']} → {data_b['column_count']} columns 
                        <span class="{'diff-positive' if col_diff > 0 else 'diff-negative' if col_diff < 0 else 'diff-neutral'}">
                            ({col_diff:+})
                        </span>
                    </li>
                    <li><strong>Completeness:</strong> {data_a['data_completeness']:.1f}% → {data_b['data_completeness']:.1f}% 
                        <span class="{'diff-positive' if completeness_diff > 0 else 'diff-negative' if completeness_diff < 0 else 'diff-neutral'}">
                            ({completeness_diff:+.1f}%)
                        </span>
                    </li>
                </ul>
            </div>
            
            <div class="comparison-grid">
                <div class="version-card">
                    <h3>📋 Version {version_a}</h3>
                    <div class="metric"><strong>Timestamp:</strong> {data_a['timestamp']}</div>
                    <div class="metric"><strong>Quality Score:</strong> {data_a['overall_quality_score']:.1f}%</div>
                    <div class="metric"><strong>Total Validations:</strong> {data_a['total_validations']}</div>
                    <div class="metric"><strong>Passed Validations:</strong> {data_a['passed_validations']}</div>
                    <div class="metric"><strong>Row Count:</strong> {data_a['row_count']:,}</div>
                    <div class="metric"><strong>Column Count:</strong> {data_a['column_count']}</div>
                    <div class="metric"><strong>Data Completeness:</strong> {data_a['data_completeness']:.1f}%</div>
                </div>
                
                <div class="version-card">
                    <h3>📋 Version {version_b}</h3>
                    <div class="metric"><strong>Timestamp:</strong> {data_b['timestamp']}</div>
                    <div class="metric"><strong>Quality Score:</strong> {data_b['overall_quality_score']:.1f}%</div>
                    <div class="metric"><strong>Total Validations:</strong> {data_b['total_validations']}</div>
                    <div class="metric"><strong>Passed Validations:</strong> {data_b['passed_validations']}</div>
                    <div class="metric"><strong>Row Count:</strong> {data_b['row_count']:,}</div>
                    <div class="metric"><strong>Column Count:</strong> {data_b['column_count']}</div>
                    <div class="metric"><strong>Data Completeness:</strong> {data_b['data_completeness']:.1f}%</div>
                </div>
            </div>
    """
    
    # Add field-level comparison if detailed stats are available
    if data_a.get('has_detailed_stats') and data_b.get('has_detailed_stats'):
        try:
            stats_a = json.loads(data_a.get('field_statistics', '{}'))
            stats_b = json.loads(data_b.get('field_statistics', '{}'))
            
            if stats_a and stats_b:
                html_content += """
                <h2>📊 Field-Level Comparison</h2>
                <table>
                    <tr>
                        <th>Field Name</th>
                        <th>Version A Type</th>
                        <th>Version B Type</th>
                        <th>Null % Change</th>
                        <th>Unique Count Change</th>
                    </tr>
                """
                
                all_fields = set(stats_a.keys()) | set(stats_b.keys())
                for field in sorted(all_fields):
                    field_a = stats_a.get(field, {})
                    field_b = stats_b.get(field, {})
                    
                    type_a = field_a.get('data_type', 'N/A')
                    type_b = field_b.get('data_type', 'N/A')
                    
                    null_a = field_a.get('null_percentage', 0)
                    null_b = field_b.get('null_percentage', 0)
                    null_diff = null_b - null_a
                    
                    unique_a = field_a.get('unique_count', 0)
                    unique_b = field_b.get('unique_count', 0)
                    unique_diff = unique_b - unique_a
                    
                    html_content += f"""
                    <tr>
                        <td>{field}</td>
                        <td>{type_a}</td>
                        <td>{type_b}</td>
                        <td class="{'diff-positive' if null_diff < 0 else 'diff-negative' if null_diff > 0 else 'diff-neutral'}">{null_diff:+.1f}%</td>
                        <td class="{'diff-positive' if unique_diff > 0 else 'diff-negative' if unique_diff < 0 else 'diff-neutral'}">{unique_diff:+}</td>
                    </tr>
                    """
                
                html_content += "</table>"
        except:
            pass  # Skip field comparison if JSON parsing fails
    
    # Add schema comparison
    schema_a = data_a.get('schema_fingerprint', '')
    schema_b = data_b.get('schema_fingerprint', '')
    
    if schema_a and schema_b:
        html_content += f"""
        <h2>🏗️ Schema Changes</h2>
        <div class="comparison-grid">
            <div class="version-card">
                <h4>Version {version_a} Schema</h4>
                <code style="font-size: 0.8em; word-break: break-all;">{schema_a}</code>
            </div>
            <div class="version-card">
                <h4>Version {version_b} Schema</h4>
                <code style="font-size: 0.8em; word-break: break-all;">{schema_b}</code>
            </div>
        </div>
        <p><strong>Schema Changed:</strong> {'Yes' if schema_a != schema_b else 'No'}</p>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    return html_content

if __name__ == "__main__":
    main()