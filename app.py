import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
# Helper Functions
# ---------------------------
def ensure_model_features(data, model):
    """Ensure all required model features exist in the dataframe."""
    missing_features = set(model.feature_names_in_) - set(data.columns)
    for feature in missing_features:
        data[feature] = 0
    # Return only the features needed by the model, in the right order
    return data[model.feature_names_in_]

def get_categorical_features():
    """Return list of categorical features that were one-hot encoded in Colab"""
    # These should match exactly what was in your Colab notebook
    return [
        "Sillhouette", "Print", "Print Technique", "Color Combo",
        "Fabric - Top", "Fabric - Bottom", "Fabric - Full Garment",
        "Gender", "Category", "Sub Category", "Collection No", "Style Number", "Size"
    ]

def encode_for_prediction(row_data, model):
    """Dynamically encode a single row for prediction"""
    # Create a DataFrame with this single row
    row_df = pd.DataFrame([row_data])
    
    # Get categorical columns that need encoding
    cat_features = get_categorical_features()
    present_cat_features = [col for col in cat_features if col in row_df.columns]
    
    # One-hot encode the categorical features
    encoded_df = pd.get_dummies(row_df, columns=present_cat_features, drop_first=True)
    
    # Ensure all required features exist
    result_df = ensure_model_features(encoded_df, model)
    
    return result_df

# ---------------------------
# Load the Model
# ---------------------------
@st.cache_resource
def load_model():
    try:
        with open('model/my_xgb_model.pkl', 'rb') as file:
            model = pickle.load(file)
            
        # Validate model
        if not hasattr(model, 'feature_names_in_'):
            st.error("Model appears invalid - missing feature_names_in_")
            return None
            
        # Show model info in debug mode
        if st.session_state.get('debug_mode', False):
            st.write(f"Model type: {type(model).__name__}")
            st.write(f"Number of features: {len(model.feature_names_in_)}")
            
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'model/my_xgb_model.pkl' exists.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ---------------------------
# Load Full Data (Original Only)
# ---------------------------   
@st.cache_resource
def load_full_data():
    try:
        # Load only the original dataset - we'll encode on-the-fly
        df = pd.read_csv('data/df.csv')
        
        # Only strip whitespace like in the original notebook
        df.columns = df.columns.str.strip()
        
        # Verify we have the lag features needed for prediction
        required_features = ['lag_1', 'lag_2', 'lag_3']
        missing_req = [f for f in required_features if f not in df.columns]
        if missing_req:
            st.warning(f"Missing required lag features in data: {missing_req}")
        
        return df
    except FileNotFoundError:
        st.error("Data file not found. Please make sure 'data/df.csv' exists.")
        return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🧵 Stock Allocation Model")
st.write("Predict allocation quantities for the next 3 months based on historical data patterns")

# Initialize session state
if "rerun" not in st.session_state:
    st.session_state.rerun = False
    
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

# Create sidebar for configuration and help
with st.sidebar:
    st.title("Configuration")
    
    # Model info
    st.subheader("Model Information")
    st.info("""
    This application uses an XGBoost model to predict stock allocation
    for the next 3 months (February, March, April) based on historical data patterns.
    """)
    
    # Configuration options
    st.subheader("Display Options")
    st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
    force_debug = st.checkbox("Force Verbose Debugging", value=False)
    show_details = st.checkbox("Show Model Features", value=False)
    
    if force_debug:
        st.session_state.debug_mode = True
    
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
    st.sidebar.caption("Quantity Allocation App v1.0")

# Load model and data
with st.spinner("Loading model and data..."):
    model = load_model()
    df = load_full_data()

if model is None or df is None:
    st.error("Failed to load required files. Please check your setup.")
    st.stop()

# Test model with dummy data (debugging)
if model is not None and st.session_state.debug_mode:
    st.write("### Model Test")
    # Create dummy data with correct feature names
    dummy_features = pd.DataFrame({name: [0.0] for name in model.feature_names_in_})
    try:
        prediction = model.predict(dummy_features)
        st.success(f"Test prediction with zeros: {prediction[0]}")
    except Exception as e:
        st.error(f"Model test failed: {str(e)}")

# Display model features if checkbox is selected
if show_details and model is not None:
    st.write("### Model Features:")
    st.write(model.feature_names_in_)

# Define available style numbers (from original dataset)
available_styles = sorted(df['Style Number'].dropna().unique())

# Allow user to select styles
selected_styles = st.multiselect(
    "Select Style Numbers:",
    available_styles,
    default=[
        "C17-INWR-02", "C20-INWR-02", "C21-INWR-04", "C20-INWR-08",
        "C17-INWR-07", "C16-INWR-23", "C20-INWR-17", "C17-INWR-17",
        "C14-INWR-19", "C16-INWR-16", "C16-INWR-18", "C15-INWR-19",
        "C20-INWR-12", "C20-INWR-13", "C20-INWR-14", "C18-INWR-08", 
        "C17-INWR-09"
    ][:5]  # Default to first 5 styles to avoid overwhelming the UI
)

if selected_styles:
    with st.spinner("Generating predictions..."):
        # Fix type mismatch (ensure consistent type)
        selected_styles = [str(s) for s in selected_styles]
    
        # Filter data based on selected styles
        selected_data = df[df['Style Number'].isin(selected_styles)]
        
        if selected_data.empty:
            st.warning("No data found for the selected style numbers.")
            st.stop()
            
        # Get last observation for each selected style - note capitalized ITEM
        last_obs = selected_data.groupby(['ITEM', 'Platform']).tail(1).copy()
    
        if not last_obs.empty:
            # Add debugging information
            if st.session_state.debug_mode:
                st.write("### Debugging Info")
                st.write("Model features:", model.feature_names_in_)
                st.write("Available columns in data:", last_obs.columns.tolist())
            
            # Prediction logic using dynamic encoding
            predictions = {}
            for idx, row in last_obs.iterrows():
                current_state = row.copy()
                monthly_preds = []
                
                for i in range(3):  # Predict next 3 months
                    try:
                        # Dynamically encode this row
                        X_sample = encode_for_prediction(current_state, model)
                        
                        # Make prediction
                        pred = model.predict(X_sample)[0]
                        pred_rounded = int(round(pred))
                        monthly_preds.append(max(0, pred_rounded))
                        
                        # Update lag features - exactly as in your Colab code
                        current_state['lag_3'] = current_state['lag_2']
                        current_state['lag_2'] = current_state['lag_1']
                        current_state['lag_1'] = pred
                    except Exception as e:
                        if st.session_state.debug_mode:
                            st.error(f"Prediction error: {str(e)}")
                            st.error(f"Error details: {type(e).__name__}: {e}")
                        monthly_preds.append(0)  # Fallback to 0 if prediction fails
                
                # Store predictions for this item/platform combination
                predictions[(row['ITEM'], row['Platform'])] = monthly_preds
    
            # Create DataFrame from predictions - use uppercase ITEM like in Colab
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
            
            if st.session_state.debug_mode:
                st.write("### Raw Results")
                st.write(results_df)
                
                # Debug column names before merge
                st.write("### DataFrame Column Names")
                st.write("results_df columns:", results_df.columns.tolist())
                st.write("df columns:", df.columns.tolist())
    
            # Merge with original data for context - use correct column names
            detailed_results = pd.merge(
                results_df,
                df[df['Style Number'].isin(selected_styles)],
                on=['ITEM', 'Platform'],  # Use uppercase ITEM
                how='left'
            )
            
            if detailed_results.empty:
                st.error("No matching records found after merging. Check 'ITEM' and 'Platform' columns.")
                st.stop()
    
            # ---------------------------
            # Summary Statistics
            # ---------------------------
            st.write("### Summary Statistics")
            
            total_predicted = detailed_results['Total_3m'].sum()
            feb_total = detailed_results['Feb_pred'].sum()
            mar_total = detailed_results['Mar_pred'].sum() 
            apr_total = detailed_results['Apr_pred'].sum()
            
            # Display KPIs in columns
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            
            with kpi1:
                st.metric(label="Total 3-Month Allocation", value=f"{total_predicted:,.0f}")
                
            with kpi2:
                st.metric(label="February", value=f"{feb_total:,.0f}")
                
            with kpi3:
                st.metric(label="March", value=f"{mar_total:,.0f}")
                
            with kpi4:
                st.metric(label="April", value=f"{apr_total:,.0f}")
                
            # Monthly trend calculation
            month_trend = ((apr_total - feb_total) / feb_total) * 100 if feb_total > 0 else 0
            st.metric(
                label="3-Month Trend (Feb to Apr)", 
                value=f"{month_trend:.1f}%",
                delta=f"{month_trend:.1f}%" if month_trend != 0 else None
            )
    
            # ---------------------------
            # Filter and Aggregation Options
            # ---------------------------
            st.write("### Filter and Aggregate Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Filter by platform - note capital P in Platform
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
                    ["None", "Style Number", "Platform"]
                )
            
            # Apply filters - note capital P in Platform
            if selected_platforms:
                filtered_results = detailed_results[detailed_results['Platform'].isin(selected_platforms)]
            else:
                filtered_results = detailed_results
                
            # Apply aggregation - use exact column names
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
    
            # ---------------------------
            # Display Results
            # ---------------------------
            st.write("### Detailed Prediction Results")
            
            # Get columns that exist in filtered_results, use correct case
            display_cols = ['ITEM', 'Style Number', 'Platform', 'Feb_pred', 'Mar_pred', 'Apr_pred', 'Total_3m']
            display_cols = [col for col in display_cols if col in filtered_results.columns]
            
            st.dataframe(filtered_results[display_cols])
    
            # ---------------------------
            # Stock Analysis (optional)
            # ---------------------------
            if 'GRN Qty' in filtered_results.columns and 'QTY SOLD' in filtered_results.columns:
                with st.expander("Stock Analysis"):
                    # Calculate current stock
                    filtered_results['Current_Stock'] = filtered_results['GRN Qty'] - filtered_results['QTY SOLD']
                    filtered_results['Current_Stock'] = filtered_results['Current_Stock'].clip(lower=0)
                    
                    # Calculate monthly average
                    filtered_results['Monthly_Avg'] = filtered_results['Total_3m'] / 3
                    
                    # Calculate weeks of inventory
                    filtered_results['Weeks_of_Inventory'] = (filtered_results['Current_Stock'] / (filtered_results['Monthly_Avg']/4)).round(1)
                    filtered_results['Weeks_of_Inventory'] = filtered_results['Weeks_of_Inventory'].replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    # Define stockout risk
                    conditions = [
                        (filtered_results['Weeks_of_Inventory'] < 4),
                        (filtered_results['Weeks_of_Inventory'] < 8)
                    ]
                    choices = ['High', 'Medium']
                    filtered_results['Stockout_Risk'] = np.select(conditions, choices, default='Low')
                    
                    st.write("### Stock Analysis")
                    st.dataframe(filtered_results[['ITEM', 'Style Number', 'Platform', 'Current_Stock', 
                                                 'Monthly_Avg', 'Weeks_of_Inventory', 'Stockout_Risk']])
    
            # ---------------------------
            # Plot Results
            # ---------------------------
            st.write("### Forecast Charts")
            
            tab1, tab2 = st.tabs(["By Style", "By Month"])
            
            with tab1:
                # Style-based chart
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                
                # Check if there are too many styles to plot effectively
                if len(filtered_results['Style Number'].unique()) > 15:
                    st.warning("Too many styles selected. Showing aggregated view only.")
                    # Aggregate by style number
                    pivot_data = filtered_results.groupby('Style Number').agg({
                        'Feb_pred': 'sum',
                        'Mar_pred': 'sum',
                        'Apr_pred': 'sum'
                    }).sort_values(by='Total_3m', ascending=False).head(15)  # Show top 15
                else:
                    pivot_data = filtered_results.pivot_table(
                        index='Style Number', 
                        values=['Feb_pred', 'Mar_pred', 'Apr_pred'],
                        aggfunc='sum'
                    )
                
                pivot_data.plot(kind='bar', ax=ax1)
                ax1.set_ylabel("Predicted Quantity")
                ax1.set_title("3-Month Forecast by Style Number")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig1)
                
            with tab2:
                # Month-based chart
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                monthly_totals = [
                    filtered_results['Feb_pred'].sum(),
                    filtered_results['Mar_pred'].sum(),
                    filtered_results['Apr_pred'].sum()
                ]
                ax2.bar(['February', 'March', 'April'], monthly_totals)
                ax2.set_ylabel("Total Quantity")
                ax2.set_title("Total Monthly Allocation")
                plt.tight_layout()
                st.pyplot(fig2)
    
            # ---------------------------
            # Download Results
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

# Refresh button
if st.button("Refresh Predictions"):
    st.session_state.rerun = True
    st.experimental_rerun()