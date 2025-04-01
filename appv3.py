import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from typing import Dict, List, Tuple, Optional

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Stock Allocation Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Load Model Package
# ---------------------------
@st.cache_resource
def load_model_package():
    try:
        # Load the complete model package
        package = joblib.load('model/model_package.pkl')
        
        model = package['model']
        sample_data = package['sample_data']
        feature_names = package['feature_names']
        categorical_features = package['categorical_features']
        features = package['features']
        X_dtypes = package['X_dtypes']
        
        return model, sample_data, feature_names, categorical_features, features, X_dtypes
    except Exception as e:
        st.error(f"Error loading model package: {str(e)}")
        return None, None, None, None, None, None

# ---------------------------
# Data Processing Functions
# ---------------------------
def preprocess_data(df):
    """Create lag features just like in the Colab notebook"""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Sort by ITEM, Platform, Year, Month
    df = df.sort_values(by=["ITEM", "Platform", "Year", "Month"])
    
    # Create lag features by grouping on (ITEM, Platform) and shifting QTY SOLD
    df['lag_1'] = df.groupby(['ITEM','Platform'])['QTY SOLD'].shift(1).fillna(0)
    df['lag_2'] = df.groupby(['ITEM','Platform'])['QTY SOLD'].shift(2).fillna(0)
    df['lag_3'] = df.groupby(['ITEM','Platform'])['QTY SOLD'].shift(3).fillna(0)
    
    # Create target_3m (sum of next 3 months), fill missing with 0
    df['target_3m'] = (
        df.groupby(['ITEM','Platform'])['QTY SOLD'].shift(-1).fillna(0) +
        df.groupby(['ITEM','Platform'])['QTY SOLD'].shift(-2).fillna(0) +
        df.groupby(['ITEM','Platform'])['QTY SOLD'].shift(-3).fillna(0)
    )
    
    return df

@st.cache_data
def load_data(data_path: str):
    """Load the dataset (original non-encoded data)"""
    try:
        # Try Excel first (preferred)
        try:
            df = pd.read_excel(data_path)
            source = "Excel"
        except:
            # Fall back to CSV
            df = pd.read_csv(data_path)
            source = "CSV"
            
        # Clean column names (match Colab preprocessing)
        df.columns = df.columns.str.strip()
        
        # IMPORTANT: Create lag features (like in Colab)
        df = preprocess_data(df)
        
        if st.session_state.debug_mode:
            st.write(f"Loaded data from {source} file")
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ---------------------------
# Prediction Function Using Template - IMPROVED VERSION
# ---------------------------
def predict_using_template(row_data, model, sample_data, feature_names, X_dtypes, calibration_factor=0.8):
    """Use the sample data from Colab as a template for prediction - STRICT MATCHING"""
    try:
        # Create a new DataFrame based on the sample template
        prediction_row = sample_data.iloc[0:1].copy()
        
        # IMPORTANT: Instead of setting all values to 0, keep the original values
        # This preserves the one-hot encoding patterns from your Colab environment
        
        # Just update the style number columns - ZERO OUT ALL STYLE COLUMNS FIRST
        style_cols = [c for c in prediction_row.columns if c.startswith('Style Number_')]
        for col in style_cols:
            prediction_row[col] = 0
            
        # Now set the correct style column to 1
        if 'Style Number' in row_data:
            style_col = f"Style Number_{row_data['Style Number']}"
            if style_col in prediction_row.columns:
                prediction_row[style_col] = 1
                if st.session_state.debug_mode:
                    st.write(f"Set column {style_col} to 1")
            else:
                # If specific style is not in training data, find closest match
                if st.session_state.debug_mode:
                    st.warning(f"Style column {style_col} not found in sample data")
                
                # Find all style columns in the sample data
                if style_cols:
                    # As a fallback, set the first style column to 1
                    prediction_row[style_cols[0]] = 1
                    if st.session_state.debug_mode:
                        st.write(f"Using fallback style column: {style_cols[0]}")
        
        # Only update the numeric features and lag values, leave other categorical encodings as-is
        for col in ['lag_1', 'lag_2', 'lag_3', 'Month', 'Year', 'Retail Price']:
            if col in row_data and col in prediction_row.columns:
                prediction_row[col] = row_data[col]
                if st.session_state.debug_mode and col == 'lag_1':
                    st.write(f"lag_1 value: {row_data[col]}")
        
        # Force the use of only the exact feature columns used in training
        # This ensures matching column order and selection
        if feature_names is not None:
            X_pred = prediction_row[feature_names]
        else:
            X_pred = prediction_row
        
        # Match data types exactly
        X_pred = X_pred.astype(X_dtypes)
        
        # Make prediction with the template
        pred = model.predict(X_pred)[0]
        
        # Apply calibration factor to match Colab results
        pred = pred * calibration_factor
        
        if st.session_state.debug_mode:
            st.write(f"Raw prediction before calibration: {pred/calibration_factor:.2f}")
            st.write(f"Calibrated prediction: {pred:.2f}")
        
        return max(0, pred)  # Ensure non-negative
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        if st.session_state.debug_mode:
            st.write("Row data:", row_data)
            st.write("Sample template columns:", sample_data.columns[:10])
        return 0

# ---------------------------
# Get Last Observations for Selected Styles
# ---------------------------
def get_last_observations(df: pd.DataFrame, selected_styles: List[str]):
    """Get the last observation for each style & platform"""
    try:
        # Ensure the lag columns exist in the dataframe
        required_columns = ['lag_1', 'lag_2', 'lag_3']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataframe. Make sure preprocessing was done correctly.")
        
        # Create an item-details subset from the original df (exactly as in Colab)
        df_item_details = df[[
            "ITEM", "Platform", "Collection No", "Style Number", "Size", "Item Name",
            "Category", "Sillhouette", "Fabric - Top", "Fabric - Bottom", 
            "Fabric - Full Garment", "Color Combo"
        ]]
        
        # Filter to only records whose "Style Number" is in selected_styles
        desired_details = df_item_details[df_item_details["Style Number"].isin(selected_styles)]
        
        # Get all lag features needed for prediction
        lag_features = ['lag_1', 'lag_2', 'lag_3', 'Month', 'Year', 'Retail Price']
        
        # Create a merged dataset that has both item details and lag features
        result = pd.merge(
            desired_details,
            df[lag_features + ['ITEM', 'Platform']],
            on=['ITEM', 'Platform'],
            how='inner'
        )
        
        # Get the last observation for each (ITEM, Platform) among the desired set
        last_obs = result.groupby(['ITEM','Platform']).tail(1).copy()
        
        return last_obs, desired_details
    
    except Exception as e:
        st.error(f"Error getting last observations: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# ---------------------------
# Process Predictions
# ---------------------------
def process_predictions(predictions: Dict[Tuple[str, str], List[float]]) -> pd.DataFrame:
    """Process predictions to match Colab output format"""
    results = []
    for (item, platform), preds in predictions.items():
        results.append({
            "ITEM": item,
            "Platform": platform,
            "Feb_pred": preds[0],
            "Mar_pred": preds[1],
            "Apr_pred": preds[2],
            "Total_3m": sum(preds)
        })
    
    results_df = pd.DataFrame(results)
    
    # Round and clip prediction columns exactly as in Colab
    for col in ["Feb_pred", "Mar_pred", "Apr_pred", "Total_3m"]:
        if col in results_df.columns:
            results_df[col] = results_df[col].round(0).clip(lower=0).astype(int)
    
    return results_df




# ---------------------------
# Model Validation Functions
# ---------------------------
def calculate_rmse(df, model, sample_data, feature_names, X_dtypes, calibration_factor):
    """Calculate RMSE (Root Mean Square Error) for model validation"""
    # Find records where we have actual target_3m values (not zero)
    validation_data = df[df['target_3m'] > 0].copy()
    
    if len(validation_data) == 0:
        return {"message": "No validation data available with actual values"}
    
    # Limit validation sample size for performance
    validation_sample = validation_data.sample(min(100, len(validation_data)))
    
    # Make predictions and compare to actuals
    squared_errors = []
    predictions = []
    actuals = []
    
    for idx, row in validation_sample.iterrows():
        # Make prediction
        pred = predict_using_template(
            row, model, sample_data, feature_names, X_dtypes, calibration_factor
        )
        
        # Get actual value
        actual = row['target_3m']
        
        # Calculate squared error
        squared_error = (pred - actual) ** 2
        
        squared_errors.append(squared_error)
        predictions.append(pred)
        actuals.append(actual)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(squared_errors))
    
    # Calculate other metrics
    mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
    
    return {
        "rmse": rmse,
        "mae": mae,
        "sample_size": len(predictions),
        "predictions": predictions,
        "actuals": actuals
    }




# ---------------------------
# Visualization Functions
# ---------------------------
def plot_stacked_bar_chart(style_sums):
    """Create stacked bar chart for predictions"""
    # Create a label combining Style Number and Color Combo
    style_sums["Label"] = style_sums["Style Number"] + " (" + style_sums["Color Combo"] + ")"
    
    # Create the chart
    fig, ax = plt.subplots(figsize=(12,6))
    
    x = np.arange(len(style_sums))
    width = 0.8  # full width for each bar
    
    bar1 = ax.bar(x, style_sums['Feb_pred'], width, label='February')
    bar2 = ax.bar(x, style_sums['Mar_pred'], width,
               bottom=style_sums['Feb_pred'], label='March')
    bar3 = ax.bar(x, style_sums['Apr_pred'], width,
               bottom=style_sums['Feb_pred']+style_sums['Mar_pred'], label='April')
    
    ax.set_xlabel("Style Number (Color Combo)")
    ax.set_ylabel("Predicted Sales (Next 3 Months)")
    ax.set_title("Aggregated Forecast by Style Number and Color Combo (Feb, Mar, Apr)")
    ax.set_xticks(x)
    ax.set_xticklabels(style_sums['Label'], rotation=45)
    ax.legend()
    
    # Annotate each segment with its numeric value
    for i, (feb, mar, apr, total) in enumerate(zip(style_sums['Feb_pred'],
                                               style_sums['Mar_pred'],
                                               style_sums['Apr_pred'],
                                               style_sums['Total_3m'])):
        plt.text(x[i], feb/2, f"{feb}", ha='center', va='center', color='white', fontsize=9)
        plt.text(x[i], feb + mar/2, f"{mar}", ha='center', va='center', color='white', fontsize=9)
        plt.text(x[i], feb + mar + apr/2, f"{apr}", ha='center', va='center', color='white', fontsize=9)
        plt.text(x[i], total + 0.05*total, f"{total}", ha='center', va='bottom', color='black', fontsize=9)
    
    plt.tight_layout()
    
    return fig

def create_trend_chart(df, selected_styles):
    """Create historical sales trend chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for style in selected_styles:
        # Filter the DataFrame for the current style
        temp = df[df["Style Number"] == style].copy()
        
        if not temp.empty:
            # Group by Year and Month to sum up QTY SOLD
            trend = temp.groupby(["Year", "Month"], as_index=False)["QTY SOLD"].sum()
            
            # Create a Time label
            trend["Time"] = trend["Year"].astype(str) + "-" + trend["Month"].astype(str)
            
            # Plot the line for this style
            ax.plot(trend["Time"], trend["QTY SOLD"], marker="o", label=f"Style {style}")
    
    ax.set_xlabel("Time (Year-Month)")
    ax.set_ylabel("Total QTY SOLD")
    ax.set_title("Historical Sales Trend for Selected Styles")
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    return fig

# ---------------------------
# Initialize Session State
# ---------------------------
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

if "rerun" not in st.session_state:
    st.session_state.rerun = False

# ---------------------------
# Sidebar Configuration
# ---------------------------
with st.sidebar:
    st.title("Configuration")
    
    # Model info
    st.subheader("Model Information")
    st.info("""
    This application uses an XGBoost model to predict stock allocation
    for the next 3 months (February, March, April) based on historical data patterns.
    """)
    
    # Data source
    data_file = st.selectbox(
        "Select data source:", 
        ["data/df.csv"],
        index=0
    )
    
    # NEW: Add calibration factor slider
    st.subheader("Prediction Settings")
    calibration_factor = st.slider(
        "Prediction calibration factor", 
        min_value=0.4, 
        max_value=1.0, 
        value=0.62,  # Default to 0.8 (20% reduction)
        step=0.02,
        help="Adjust this to match predictions with Colab results (lower = smaller predictions)"
    )
    
    # Configuration options
    st.subheader("Display Options")
    st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
    show_trend = st.checkbox("Show Historical Trend", value=True)
    
    # Help section
    st.subheader("Help")
    with st.expander("How to use this app"):
        st.write("""
        1. Select one or more style numbers from the dropdown
        2. View the predicted allocations for the next 3 months
        3. Use the filtering options to focus on specific platforms
        4. Download the results for further analysis
        """)
    
    # Add version info
    st.sidebar.markdown("---")
    st.sidebar.caption("Quantity Allocation App v2.0")

# ---------------------------
# Main UI
# ---------------------------
st.title("🧵 Stock Allocation Model")
st.write("Predict allocation quantities for the next 3 months based on historical data patterns")

# ---------------------------
# Load Model & Data
# ---------------------------
with st.spinner("Loading model and data..."):
    # Load model package
    model, sample_data, feature_names, categorical_features, features, X_dtypes = load_model_package()
    
    # Load data file
    df = load_data(data_file)
    
    if model is None or df is None:
        st.error("Failed to load model or data files. Please check your setup.")
        st.stop()
    
    if st.session_state.debug_mode:
        st.write("### Data Overview")
        st.write(f"Total records: {len(df)}")
        st.write(f"Columns available: {', '.join(df.columns[:10])}...")
        if 'lag_1' in df.columns:
            st.success("Lag features created successfully")
        else:
            st.error("Lag features missing")
        
        # Check style columns in sample data
        style_cols = [c for c in sample_data.columns if c.startswith('Style Number_')]
        st.write(f"Found {len(style_cols)} style number columns in sample data")
        if style_cols:
            st.write(f"Example style columns: {style_cols[:5]}")
        
        # NEW: Add feature comparison button
        if st.button("Compare Feature Values"):
            st.write("### Feature Value Comparison")
            
            # Display sample value from template
            sample_values = sample_data.iloc[0:1][['lag_1', 'lag_2', 'lag_3']]
            st.write("Sample template lag values:", sample_values.values.tolist()[0])
            
            # Display some column stats
            st.write(f"Feature names length: {len(feature_names)}")
            st.write(f"First 5 feature names: {feature_names[:5]}")

# ---------------------------
# Style Selection
# ---------------------------
# Define available style numbers from the dataset
available_styles = sorted(df['Style Number'].dropna().unique())

# Default style list from Colab
default_styles = [
    "C17-INWR-02", "C20-INWR-02", "C21-INWR-04", "C20-INWR-08",
    "C17-INWR-07", "C16-INWR-23", "C20-INWR-17", "C17-INWR-17",
    "C14-INWR-19", "C16-INWR-16", "C16-INWR-18", "C15-INWR-19",
    "C20-INWR-12", "C20-INWR-13", "C20-INWR-14", "C18-INWR-08",
    "C17-INWR-09"
]

# Filter default styles to only those in dataset
default_styles = [s for s in default_styles if s in available_styles]

# Allow user to select styles
selected_styles = st.multiselect(
    "Select Style Numbers:",
    available_styles,
    default=default_styles[:5] if default_styles else available_styles[:5]  # Use first 5 by default
)

# ---------------------------
# Generate Predictions
# ---------------------------
if selected_styles:
    with st.spinner("Generating predictions - this may take a moment..."):
        # Get last observations for selected styles
        last_obs, desired_details = get_last_observations(df, selected_styles)
        
        if last_obs.empty:
            st.warning("No data found for the selected style numbers.")
            st.stop()
            
        # Debug data
        if st.session_state.debug_mode:
            st.write("### Data Debug Info")
            st.write(f"Number of observations for prediction: {len(last_obs)}")
            st.write(f"Sample data columns: {sample_data.columns[:5].tolist()}")
            st.write(f"Last observation columns: {last_obs.columns[:5].tolist()}")
        
        # Generate predictions using template approach
        predictions = {}
        for idx, row in last_obs.iterrows():
            current_state = row.copy()
            monthly_preds = []
            
            for i in range(3):  # 3 months
                try:
                    # Use template-based prediction with calibration factor
                    pred = predict_using_template(
                        current_state, 
                        model, 
                        sample_data, 
                        feature_names, 
                        X_dtypes,
                        calibration_factor
                    )
                    monthly_preds.append(pred)
                    
                    # Update lag features for next month prediction
                    current_state['lag_3'] = current_state['lag_2']
                    current_state['lag_2'] = current_state['lag_1']
                    current_state['lag_1'] = pred
                    
                except Exception as e:
                    if st.session_state.debug_mode:
                        st.error(f"Prediction error for item {current_state.get('ITEM', 'unknown')}: {e}")
                    monthly_preds.append(0)  # Fallback
                    
            # Store predictions for this item
            predictions[(row['ITEM'], row['Platform'])] = monthly_preds
        
        # Process predictions to match Colab format
        results_df = process_predictions(predictions)
        
        # Merge with desired details to get style info
        detailed_results = pd.merge(
            results_df,
            desired_details,
            on=["ITEM", "Platform"],
            how="left"
        )
        
        if detailed_results.empty:
            st.error("No matching records found after merging.")
            st.stop()
        
        # ---------------------------
        # Summary Statistics
        # ---------------------------
        st.write("### Summary Statistics")
        
        total_predicted = detailed_results['Total_3m'].sum()
        feb_total = detailed_results['Feb_pred'].sum()
        mar_total = detailed_results['Mar_pred'].sum() 
        apr_total = detailed_results['Apr_pred'].sum()
        
        # Display KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.metric(label="Total 3-Month Allocation", value=f"{total_predicted:,.0f}")
            
        with kpi2:
            st.metric(label="February", value=f"{feb_total:,.0f}")
            
        with kpi3:
            st.metric(label="March", value=f"{mar_total:,.0f}")
            
        with kpi4:
            st.metric(label="April", value=f"{apr_total:,.0f}")



        # ---------------------------
        # Model Validation
        # ---------------------------
        st.write("### Model Validation")
        
        with st.spinner("Calculating model accuracy..."):
            validation_results = calculate_rmse(
                df, model, sample_data, feature_names, X_dtypes, calibration_factor
            )
            
            if "rmse" in validation_results:
                # Create columns for metrics
                metric1, metric2, metric3 = st.columns(3)
                
                with metric1:
                    st.metric("RMSE (Root Mean Square Error)", f"{validation_results['rmse']:.2f}")
                    
                with metric2:
                    st.metric("MAE (Mean Absolute Error)", f"{validation_results['mae']:.2f}")
                    
                with metric3:
                    st.metric("Sample Size", validation_results['sample_size'])
                
                # Add a visual comparison if user wants to see it
                if st.checkbox("Show Prediction vs Actual Plot"):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(validation_results['actuals'], validation_results['predictions'], alpha=0.5)
                    
                    # Add diagonal line (perfect predictions)
                    max_val = max(max(validation_results['actuals']), max(validation_results['predictions']))
                    ax.plot([0, max_val], [0, max_val], 'r--')
                    
                    ax.set_xlabel("Actual Values")
                    ax.set_ylabel("Predicted Values")
                    ax.set_title("Model Validation: Predicted vs Actual Values")
                    st.pyplot(fig)
                    
                    st.caption("Points on or near the red line indicate accurate predictions.")
            else:
                st.info("Validation could not be performed. No historical data with known outcomes available.")
        
        # ---------------------------
        # Filter and Aggregation Options
        # ---------------------------
        st.write("### Filter and Aggregate Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Filter by platform
            platforms = sorted(detailed_results['Platform'].unique())
            selected_platforms = st.multiselect(
                "Filter by Platform:",
                platforms,
                default=platforms
            )
        
        with col2:
            # Aggregation options
            aggregation = st.selectbox(
                "Aggregate by:",
                ["None", "Style Number", "Platform", "Style Number and Color Combo"]
            )
        
        # Apply filters
        if selected_platforms:
            filtered_results = detailed_results[detailed_results['Platform'].isin(selected_platforms)]
        else:
            filtered_results = detailed_results
        
        # Apply aggregation
        if aggregation == "Style Number":
            agg_results = filtered_results.groupby('Style Number').agg({
                'Feb_pred': 'sum',
                'Mar_pred': 'sum',
                'Apr_pred': 'sum',
                'Total_3m': 'sum'
            }).reset_index()
            st.write("### Aggregated by Style Number:")
            st.dataframe(agg_results)
            
        elif aggregation == "Platform":
            agg_results = filtered_results.groupby('Platform').agg({
                'Feb_pred': 'sum',
                'Mar_pred': 'sum',
                'Apr_pred': 'sum',
                'Total_3m': 'sum'
            }).reset_index()
            st.write("### Aggregated by Platform:")
            st.dataframe(agg_results)
            
        elif aggregation == "Style Number and Color Combo" and 'Color Combo' in filtered_results.columns:
            agg_results = filtered_results.groupby(['Style Number', 'Color Combo']).agg({
                'Feb_pred': 'sum',
                'Mar_pred': 'sum',
                'Apr_pred': 'sum',
                'Total_3m': 'sum'
            }).reset_index()
            st.write("### Aggregated by Style Number and Color Combo:")
            st.dataframe(agg_results)
        
        # ---------------------------
        # Display Results
        # ---------------------------
        st.write("### Detailed Prediction Results")
        
        # Select display columns that exist
        display_cols = [
            "ITEM", "Item Name", "Collection No", "Style Number", "Size", "Platform",
            "Sillhouette", "Fabric - Top", "Fabric - Bottom", "Fabric - Full Garment",
            "Feb_pred", "Mar_pred", "Apr_pred", "Total_3m"
        ]
        display_cols = [col for col in display_cols if col in filtered_results.columns]
        st.dataframe(filtered_results[display_cols])
        
        # ---------------------------
        # Visualization - Stacked Bar Chart
        # ---------------------------
        st.write("### Visualizations")
        
        tab1, tab2 = st.tabs(["Forecast by Style & Color", "Historical Trend"])
        
        with tab1:
            # Group the detailed forecast results by "Style Number" and "Color Combo"
            if 'Color Combo' in filtered_results.columns:
                style_sums = filtered_results.groupby(["Style Number", "Color Combo"], as_index=False).agg({
                    "Feb_pred": "sum",
                    "Mar_pred": "sum",
                    "Apr_pred": "sum"
                })
                style_sums["Total_3m"] = (style_sums["Feb_pred"] + 
                                        style_sums["Mar_pred"] + 
                                        style_sums["Apr_pred"])
                
                # Display the stacked bar chart
                st.pyplot(plot_stacked_bar_chart(style_sums))
            else:
                st.warning("Color Combo column not found in data. Cannot create style/color visualization.")
        
        with tab2:
            if show_trend:
                # Show historical sales trend
                st.pyplot(create_trend_chart(df, selected_styles))
            else:
                st.info("Enable 'Show Historical Trend' in the sidebar to see historical sales data")
        
        # ---------------------------
        # Download Options
        # ---------------------------
        st.write("### Download Options")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Download Detailed Results",
                data=filtered_results.to_csv(index=False).encode('utf-8'),
                file_name="predicted_results_detailed.csv",
                mime='text/csv'
            )
        
        with col2:
            if aggregation != "None":
                st.download_button(
                    label=f"Download Aggregated Results ({aggregation})",
                    data=agg_results.to_csv(index=False).encode('utf-8'),
                    file_name=f"predicted_results_{aggregation.lower().replace(' ', '_')}.csv",
                    mime='text/csv'
                )

# ---------------------------
# Refresh Button
# ---------------------------
if st.button("Refresh Predictions"):
    st.session_state.rerun = True
    st.experimental_rerun()