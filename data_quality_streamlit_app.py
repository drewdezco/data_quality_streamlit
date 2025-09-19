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
    page_title="Data Quality Validation Service",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
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
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Data Quality Validation Service</h1>
        <p>Comprehensive data validation and quality reporting for your CSV datasets</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("🔧 Configuration")
    
    # Service information
    with st.sidebar.expander("ℹ️ How to Use", expanded=False):
        st.markdown("""
        **Simple 4-Step Process:**
        1. **Upload CSV** - Drag & drop your file
        2. **Configure Validations** *(Optional)*
        3. **Mark Critical Columns** *(Optional)*  
        4. **Generate Report** - Get comprehensive analysis
        
        **What You Get:**
        - Interactive HTML reports
        - Data quality metrics
        - Field-by-field analysis
        - Professional charts and visualizations
        """)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Upload Data", "✅ Configure Validations", "⭐ Critical Columns", "📊 Generate Report", "📈 Historical Tracking"])
    
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

    # Tab 1: Upload Data
    with tab1:
        st.header("📁 Upload Your CSV File")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
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
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>💡 Tips for Best Results</h3>
                <ul>
                    <li>Ensure first row contains column headers</li>
                    <li>Use consistent date formats</li>
                    <li>Avoid special characters in column names</li>
                    <li>Check for extra commas or quotes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

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
            
            # Validation type selection
            st.subheader("🔧 Select Validation Types")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Not Null validations
                st.markdown("**🚫 Not Null Checks**")
                not_null_columns = st.multiselect(
                    "Select columns that should never be empty:",
                    options=df.columns.tolist(),
                    key="not_null_cols",
                    help="These columns will be checked for missing values"
                )
                
                # Uniqueness validations  
                st.markdown("**🆔 Uniqueness Checks**")
                unique_columns = st.multiselect(
                    "Select columns that should have unique values:",
                    options=df.columns.tolist(),
                    key="unique_cols",
                    help="These columns will be checked for duplicate values"
                )
                
                # Range validations
                st.markdown("**📊 Range Validations**")
                range_validations = {}
                for col in df.select_dtypes(include=['number']).columns:
                    if st.checkbox(f"Set range for {col}", key=f"range_{col}"):
                        col_min, col_max = st.columns(2)
                        with col_min:
                            min_val = st.number_input(f"Min value for {col}", key=f"min_{col}")
                        with col_max:
                            max_val = st.number_input(f"Max value for {col}", key=f"max_{col}")
                        range_validations[col] = (min_val, max_val)
            
            with col2:
                # Allowed values validations
                st.markdown("**📝 Allowed Values**")
                allowed_values_validations = {}
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols[:5]:  # Limit to first 5 for UI purposes
                    if st.checkbox(f"Set allowed values for {col}", key=f"allowed_{col}"):
                        unique_vals = df[col].dropna().unique()[:20]  # Show max 20 unique values
                        selected_vals = st.multiselect(
                            f"Allowed values for {col}:",
                            options=unique_vals.tolist(),
                            default=unique_vals.tolist(),
                            key=f"vals_{col}"
                        )
                        if selected_vals:
                            allowed_values_validations[col] = selected_vals
                
                # Regex pattern validations
                st.markdown("**🔍 Pattern Matching**")
                regex_validations = {}
                
                # Common patterns
                pattern_presets = {
                    "Email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
                    "Phone (US)": r'^\(\d{3}\)\s\d{3}-\d{4}$',
                    "ZIP Code": r'^\d{5}(-\d{4})?$',
                    "Custom": ""
                }
                
                for col in df.select_dtypes(include=['object']).columns[:3]:  # Limit for UI
                    if st.checkbox(f"Set pattern for {col}", key=f"pattern_{col}"):
                        pattern_type = st.selectbox(
                            f"Pattern type for {col}:",
                            options=list(pattern_presets.keys()),
                            key=f"pattern_type_{col}"
                        )
                        
                        if pattern_type == "Custom":
                            pattern = st.text_input(f"Custom regex pattern for {col}:", key=f"custom_pattern_{col}")
                        else:
                            pattern = pattern_presets[pattern_type]
                            st.code(pattern, language="regex")
                        
                        if pattern:
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
                <p>Mark your most important data fields for enhanced monitoring and prioritized analysis. Critical columns receive special attention in reports.</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Select Critical Columns")
                
                critical_columns = st.multiselect(
                    "Choose columns that are critical for your business:",
                    options=df.columns.tolist(),
                    default=st.session_state.critical_columns,
                    help="Critical columns get priority analysis and enhanced monitoring"
                )
                
                if critical_columns:
                    st.markdown("**Selected Critical Columns:**")
                    for col in critical_columns:
                        col_type = str(df[col].dtype)
                        unique_count = df[col].nunique()
                        null_count = df[col].isnull().sum()
                        
                        st.markdown(f"""
                        <div class="feature-card">
                            <strong>🔹 {col}</strong><br>
                            Type: {col_type} | Unique Values: {unique_count} | Null Values: {null_count}
                        </div>
                        """, unsafe_allow_html=True)
                
                if st.button("💾 Save Critical Columns", type="primary"):
                    st.session_state.critical_columns = critical_columns
                    st.success(f"✅ Saved {len(critical_columns)} critical columns!")
            
            with col2:
                st.markdown("""
                <div class="feature-card">
                    <h3>🌟 Benefits of Critical Columns</h3>
                    <ul>
                        <li><strong>Priority Analysis</strong> - Checked first</li>
                        <li><strong>Enhanced Validation</strong> - Additional checks</li>
                        <li><strong>Special Alerts</strong> - Higher priority issues</li>
                        <li><strong>Executive Summary</strong> - Featured in dashboard</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <h3>💡 Common Critical Columns</h3>
                    <ul>
                        <li>Primary identifiers (ID, SSN)</li>
                        <li>Contact information</li>
                        <li>Financial data</li>
                        <li>Compliance fields</li>
                        <li>Key business metrics</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # Tab 4: Generate Report
    with tab4:
        st.header("📊 Generate Data Quality Report")
        
        if st.session_state.df is None:
            st.warning("Please upload a CSV file first in the 'Upload Data' tab.")
        else:
            df = st.session_state.df
            
            # Report configuration
            col1, col2 = st.columns([2, 1])
            
            with col1:
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
                if st.button("🚀 Generate Data Quality Report", type="primary", use_container_width=True):
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
                            
                            st.success("✅ Report generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error generating report: {str(e)}")
            
            with col2:
                st.markdown("""
                <div class="feature-card">
                    <h3>📋 Report Features</h3>
                    <ul>
                        <li><strong>Overview Dashboard</strong> - Key metrics and charts</li>
                        <li><strong>Validation Results</strong> - Detailed pass/fail analysis</li>
                        <li><strong>Field Summary</strong> - Column-by-column statistics</li>
                        <li><strong>Interactive Charts</strong> - Visual data insights</li>
                        <li><strong>Export Options</strong> - CSV downloads</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <h3>⏱️ Processing Time</h3>
                    <p>Report generation typically takes:</p>
                    <ul>
                        <li>Small datasets (&lt;1K rows): ~5 seconds</li>
                        <li>Medium datasets (1K-10K rows): ~15 seconds</li>
                        <li>Large datasets (&gt;10K rows): ~30 seconds</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Display generated report
            if st.session_state.report_generated:
                st.markdown("---")
                st.subheader("📊 Your Data Quality Report")
                
                col_a, col_b, col_c = st.columns([1, 1, 1])
                
                with col_a:
                    # Download HTML report
                    html_bytes = st.session_state.report_html.encode('utf-8')
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=html_bytes,
                        file_name=f"{st.session_state.report_title.replace(' ', '_')}.html",
                        mime="text/html",
                        type="primary"
                    )
                
                with col_b:
                    # Download results as CSV
                    if st.session_state.checker is not None:
                        results_df = st.session_state.checker.get_results()
                        if not results_df.empty:
                            csv_buffer = io.StringIO()
                            results_df.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="📊 Download Results CSV",
                                data=csv_buffer.getvalue(),
                                file_name=f"validation_results_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                
                with col_c:
                    if st.button("🔄 Generate New Report"):
                        st.session_state.report_generated = False
                        st.session_state.checker = None
                        st.rerun()
                
                # Show results summary
                if st.session_state.checker is not None:
                    results_df = st.session_state.checker.get_results()
                    if not results_df.empty:
                        st.subheader("📈 Results Summary")
                        
                        # Quick metrics
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
                        st.subheader("📋 Validation Results")
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
    
    # Tab 5: Historical Tracking
    with tab5:
        st.header("📈 Historical Data Quality Tracking")
        
        # Introduction
        st.markdown("""
        <div class="step-container">
            <h3>🕒 Track Data Quality Over Time</h3>
            <p>Monitor how your data quality changes over time by maintaining a historical record of validation results. Perfect for detecting trends, quality degradation, or improvements.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Dataset versioning section
        st.subheader("📊 Dataset Version Tracking")
        
        version_col1, version_col2 = st.columns([3, 1])
        
        with version_col1:
            dataset_version = st.text_input(
                "Dataset Version/Identifier:",
                value="v1.0",
                help="Specify a version identifier for this dataset (e.g., v1.0, v2.1, Jan2024, etc.)",
                key="dataset_version"
            )
            
            st.markdown("""
            <div class="feature-card">
                <h4>💡 Version Tracking Benefits</h4>
                <p>Track multiple versions of the same dataset to:</p>
                <ul>
                    <li><strong>Compare data quality</strong> between versions</li>
                    <li><strong>Monitor schema changes</strong> (added/removed columns)</li>
                    <li><strong>Track data volume growth</strong> over versions</li>
                    <li><strong>Identify validation improvements</strong> or regressions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with version_col2:
            enable_data_export = st.checkbox(
                "📁 Export Data Snapshot", 
                value=True,
                help="Include a CSV snapshot of the current dataset in tracking"
            )
            
            enable_detailed_comparison = st.checkbox(
                "🔍 Enable Detailed Comparison",
                value=False,
                help="Track field-level statistics for version comparison"
            )
        
        st.markdown("---")
        
        # Two main sections
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Historical file upload section
            st.subheader("📂 Historical Data Management")
            
            uploaded_history = st.file_uploader(
                "Upload Historical Tracking File (Optional)",
                type=['csv'],
                help="Upload an existing historical tracking CSV to continue monitoring trends",
                key="history_upload"
            )
            
            # Version comparison section
            if st.session_state.historical_data is not None and not st.session_state.historical_data.empty:
                st.subheader("🔄 Version Comparison")
                
                # Get available versions
                available_versions = []
                if 'dataset_version' in st.session_state.historical_data.columns:
                    available_versions = sorted(st.session_state.historical_data['dataset_version'].unique().tolist())
                
                if len(available_versions) >= 2:
                    compare_col1, compare_col2 = st.columns(2)
                    
                    with compare_col1:
                        version_a = st.selectbox(
                            "Compare Version A:",
                            options=available_versions,
                            key="version_a"
                        )
                    
                    with compare_col2:
                        version_b = st.selectbox(
                            "Compare Version B:",
                            options=available_versions,
                            index=min(1, len(available_versions)-1),
                            key="version_b"
                        )
                    
                    if st.button("📊 Generate Version Comparison", help="Compare quality metrics and data statistics between versions"):
                        comparison_report = generate_version_comparison(st.session_state.historical_data, version_a, version_b)
                        
                        st.download_button(
                            label="📥 Download Version Comparison Report",
                            data=comparison_report,
                            file_name=f"version_comparison_{version_a}_vs_{version_b}_{datetime.now().strftime('%Y%m%d')}.html",
                            mime="text/html"
                        )
                        
                        # Show inline comparison
                        st.subheader(f"📊 {version_a} vs {version_b} Comparison")
                        
                        hist_df = st.session_state.historical_data
                        data_a = hist_df[hist_df['dataset_version'] == version_a].iloc[-1] if version_a in hist_df['dataset_version'].values else None
                        data_b = hist_df[hist_df['dataset_version'] == version_b].iloc[-1] if version_b in hist_df['dataset_version'].values else None
                        
                        if data_a is not None and data_b is not None:
                            comp_col1, comp_col2, comp_col3 = st.columns(3)
                            
                            with comp_col1:
                                quality_diff = data_b['overall_quality_score'] - data_a['overall_quality_score']
                                trend_icon = "↗️" if quality_diff > 0 else "↘️" if quality_diff < 0 else "➡️"
                                st.metric(
                                    "Quality Score Change",
                                    f"{data_b['overall_quality_score']:.1f}%",
                                    f"{quality_diff:+.1f}% {trend_icon}"
                                )
                            
                            with comp_col2:
                                row_diff = data_b['row_count'] - data_a['row_count']
                                st.metric(
                                    "Row Count Change",
                                    f"{data_b['row_count']:,}",
                                    f"{row_diff:+,}"
                                )
                            
                            with comp_col3:
                                val_diff = data_b['total_validations'] - data_a['total_validations']
                                st.metric(
                                    "Validation Rules Change",
                                    f"{data_b['total_validations']}",
                                    f"{val_diff:+}"
                                )
                else:
                    st.info("📊 Upload data with multiple versions to enable version comparison features.")
            
            if uploaded_history is not None:
                try:
                    historical_df = pd.read_csv(uploaded_history)
                    st.session_state.historical_data = historical_df
                    st.session_state.tracking_enabled = True
                    
                    st.markdown('<div class="success-message">✅ Historical data loaded successfully!</div>', unsafe_allow_html=True)
                    
                    # Display historical summary
                    st.subheader("📊 Historical Data Overview")
                    
                    if not historical_df.empty:
                        total_runs = len(historical_df)
                        date_range = ""
                        if 'timestamp' in historical_df.columns:
                            min_date = pd.to_datetime(historical_df['timestamp']).min().strftime('%Y-%m-%d')
                            max_date = pd.to_datetime(historical_df['timestamp']).max().strftime('%Y-%m-%d')
                            date_range = f"{min_date} to {max_date}"
                        
                        avg_score = historical_df.get('overall_quality_score', pd.Series([0])).mean()
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.markdown('<div class="metric-card"><h3>{}</h3><p>Total Runs</p></div>'.format(total_runs), unsafe_allow_html=True)
                        with col_b:
                            st.markdown('<div class="metric-card"><h3>{:.1f}%</h3><p>Avg Quality Score</p></div>'.format(avg_score), unsafe_allow_html=True)
                        with col_c:
                            st.markdown('<div class="metric-card"><h3>{}</h3><p>Days Tracked</p></div>'.format(len(historical_df['timestamp'].dt.date.unique()) if 'timestamp' in historical_df.columns else 'N/A'), unsafe_allow_html=True)
                        
                        # Show recent entries
                        st.subheader("🕒 Recent Quality Checks")
                        recent_data = historical_df.tail(10)
                        st.dataframe(recent_data, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error reading historical file: {str(e)}")
                    st.info("Please ensure your historical file has the correct format.")
            
            else:
                # Enable tracking without historical file
                if st.checkbox("🔄 Enable Historical Tracking", help="Start tracking data quality metrics from this session"):
                    st.session_state.tracking_enabled = True
                    st.info("Historical tracking enabled! Your next report generation will create the first tracking entry.")
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>📋 Historical File Format</h3>
                <p>Your historical tracking file should contain:</p>
                <ul>
                    <li><strong>timestamp</strong> - Date/time of validation</li>
                    <li><strong>dataset_name</strong> - Name of dataset</li>
                    <li><strong>overall_quality_score</strong> - Overall percentage</li>
                    <li><strong>total_validations</strong> - Number of rules</li>
                    <li><strong>passed_validations</strong> - Rules that passed</li>
                    <li><strong>row_count</strong> - Number of data rows</li>
                    <li><strong>column_count</strong> - Number of columns</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h3>🎯 Benefits of Tracking</h3>
                <ul>
                    <li><strong>Trend Analysis</strong> - See quality over time</li>
                    <li><strong>Alert Detection</strong> - Spot quality drops</li>
                    <li><strong>Compliance Reporting</strong> - Historical records</li>
                    <li><strong>Process Improvement</strong> - Track interventions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Historical analysis and charts
        if st.session_state.historical_data is not None and not st.session_state.historical_data.empty:
            st.markdown("---")
            st.subheader("📈 Historical Analysis")
            
            historical_df = st.session_state.historical_data
            
            # Ensure timestamp column is datetime
            if 'timestamp' in historical_df.columns:
                historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
                historical_df = historical_df.sort_values('timestamp')
                
                # Time-based analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    # Quality score trend
                    if 'overall_quality_score' in historical_df.columns:
                        st.markdown("**📊 Quality Score Trend**")
                        
                        # Create a simple line chart using Streamlit's built-in charting
                        chart_data = historical_df.set_index('timestamp')[['overall_quality_score']].rename(columns={'overall_quality_score': 'Quality Score %'})
                        st.line_chart(chart_data)
                        
                        # Quality statistics
                        current_score = historical_df['overall_quality_score'].iloc[-1]
                        avg_score = historical_df['overall_quality_score'].mean()
                        trend = "↗️ Improving" if len(historical_df) > 1 and historical_df['overall_quality_score'].iloc[-1] > historical_df['overall_quality_score'].iloc[-2] else "↘️ Declining" if len(historical_df) > 1 else "📊 Stable"
                        
                        st.markdown(f"""
                        <div class="feature-card">
                            <h4>Quality Metrics</h4>
                            <p><strong>Current Score:</strong> {current_score:.1f}%</p>
                            <p><strong>Average Score:</strong> {avg_score:.1f}%</p>
                            <p><strong>Trend:</strong> {trend}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Validation count trend
                    if 'total_validations' in historical_df.columns:
                        st.markdown("**✅ Validation Count Trend**")
                        
                        chart_data = historical_df.set_index('timestamp')[['total_validations']].rename(columns={'total_validations': 'Total Validations'})
                        st.line_chart(chart_data)
                        
                        # Validation statistics
                        st.markdown(f"""
                        <div class="feature-card">
                            <h4>Validation Metrics</h4>
                            <p><strong>Current Validations:</strong> {historical_df['total_validations'].iloc[-1]}</p>
                            <p><strong>Average Validations:</strong> {historical_df['total_validations'].mean():.1f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Data volume trends
                if 'row_count' in historical_df.columns:
                    st.markdown("**📊 Data Volume Trends**")
                    
                    volume_data = historical_df.set_index('timestamp')[['row_count']].rename(columns={'row_count': 'Row Count'})
                    if 'column_count' in historical_df.columns:
                        volume_data['Column Count'] = historical_df.set_index('timestamp')['column_count']
                    
                    st.line_chart(volume_data)
                
                # Historical summary table
                st.subheader("📋 Quality Summary by Period")
                
                # Group by date and show daily summaries
                if len(historical_df) > 1:
                    daily_summary = historical_df.groupby(historical_df['timestamp'].dt.date).agg({
                        'overall_quality_score': ['mean', 'min', 'max'],
                        'total_validations': 'mean',
                        'passed_validations': 'mean'
                    }).round(1)
                    
                    daily_summary.columns = ['Avg Quality %', 'Min Quality %', 'Max Quality %', 'Avg Validations', 'Avg Passed']
                    st.dataframe(daily_summary.tail(10), use_container_width=True)
        
        # Export historical data
        if st.session_state.historical_data is not None:
            st.markdown("---")
            st.subheader("📤 Export Historical Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download updated historical file
                csv_buffer = io.StringIO()
                st.session_state.historical_data.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download Historical Tracking File",
                    data=csv_buffer.getvalue(),
                    file_name=f"data_quality_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    help="Download your historical tracking data for backup or sharing"
                )
            
            with col2:
                # Generate historical report
                if st.button("📊 Generate Historical Report", help="Create a comprehensive historical analysis report"):
                    # Create a simple historical summary
                    hist_summary = generate_historical_summary(st.session_state.historical_data)
                    
                    st.download_button(
                        label="📥 Download Historical Summary",
                        data=hist_summary,
                        file_name=f"historical_summary_{datetime.now().strftime('%Y%m%d')}.html",
                        mime="text/html"
                    )
        
        # Data snapshots section
        if st.session_state.dataset_snapshots:
            st.markdown("---")
            st.subheader("📁 Dataset Snapshots")
            
            # Show available snapshots
            st.markdown("**Available Data Snapshots:**")
            
            for snapshot_id, snapshot_info in st.session_state.dataset_snapshots.items():
                with st.expander(f"📄 {snapshot_info['dataset_name']} - {snapshot_info['version']} ({snapshot_info['timestamp']})"):
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        **Dataset:** {snapshot_info['dataset_name']}  
                        **Version:** {snapshot_info['version']}  
                        **Timestamp:** {snapshot_info['timestamp']}  
                        **Full Row Count:** {snapshot_info['full_row_count']:,}  
                        **Snapshot Rows:** {len(snapshot_info['data']):,}
                        """)
                    
                    with col2:
                        # Download snapshot CSV
                        csv_buffer = io.StringIO()
                        snapshot_info['data'].to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"{snapshot_id}.csv",
                            mime="text/csv",
                            key=f"download_{snapshot_id}"
                        )
                    
                    with col3:
                        # Show preview button
                        if st.button("👀 Preview Data", key=f"preview_{snapshot_id}"):
                            st.dataframe(snapshot_info['data'].head(10), use_container_width=True)

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
