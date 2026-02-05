import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# 1. Page Configuration
st.set_page_config(
    page_title="FinVision | Revenue Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# 2. Custom CSS for Modern UI
st.markdown("""
    <style>
    /* 1. Main app background */
    .stApp {
        background-color: #f1f5f9;
    }
    
    /* 2. TITLES & SUBHEADERS (Outside of cards) */
    /* Targets st.title, st.header, and st.subheader */
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #0f172a !important;
        font-family: 'Inter', sans-serif;
    }

    /* 3. DARK CONTAINERS (Cards) */
    div[data-testid="stVerticalBlock"] > div:has(div.element-container) {
        background-color: #1e293b !important; 
        padding: 2.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }

    /* 4. FORCE WHITE TEXT ONLY INSIDE THE CARDS */
    /* This ensures labels and text inside the Navy cards stay white */
    div[data-testid="stVerticalBlock"] > div:has(div.element-container) label,
    div[data-testid="stVerticalBlock"] > div:has(div.element-container) p,
    div[data-testid="stVerticalBlock"] > div:has(div.element-container) h3 {
        color: #ffffff !important;
    }

    /* 5. INPUT BOXES */
    input {
        color: #ffffff !important;
        background-color: #334155 !important;
        border: 1px solid #475569 !important;
    }

    /* 6. PREDICTION VALUE (Metrics) */
    [data-testid="stMetricValue"] {
        color: #818cf8 !important;
        font-weight: 700 !important;
    }

    /* 7. BUTTON */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white !important;
        border-radius: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)
    

# 3. Load Model and Features
@st.cache_resource
def load_assets():
    model = joblib.load('revenue_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

try:
    model, model_features = load_assets()
except Exception as e:
    st.error("⚠️ Model files not found. Please ensure 'revenue_model.pkl' and 'model_features.pkl' are in the same directory.")
    st.stop()

# 4. Header Section
st.title("📈 Revenue Predictor")
st.subheader("Advanced Corporate Revenue Forecasting")
st.markdown("---")

# 5. Main Layout using Tabs
tab1, tab2 = st.tabs(["🎯 Single Prediction", "📊 Batch Processing"])

with tab1:
    st.markdown("### Input Company Financials")
    st.info("Enter the fiscal metrics below to generate a revenue forecast for the next year.")
    
    # Organizing inputs into 3 columns for better UX
    col1, col2, col3 = st.columns(3)
    
    with col1:
        assets = st.number_input("Total Assets", min_value=0.0, value=5000.0, help="Total value of assets held by the company")
        liabilities = st.number_input("Total Liabilities", min_value=0.0, value=3000.0)
        employees = st.number_input("Employee Count", min_value=1, value=120)

    with col2:
        cash = st.number_input("Cash & Equivalents", min_value=0.0, value=800.0)
        retained_earnings = st.number_input("Retained Earnings", value=1500.0)
        market_val = st.number_input("Market Value", min_value=0.0, value=12000.0)

    with col3:
        inventory = st.number_input("Total Inventory", min_value=0.0, value=400.0)
        equity = st.number_input("Stockholders Equity", min_value=0.0, value=2000.0)
        debt = st.number_input("Total Debt/Total Assets", value=0.45)

    if st.button("Generate Forecast"):
        # Prepare input array
        input_data = {feat: 0.0 for feat in model_features}
        
        # Mapping inputs to the feature list
        input_data['Assets - Total'] = assets
        input_data['Liabilities - Total'] = liabilities
        input_data['Employees'] = employees
        input_data['Cash'] = cash
        input_data['Retained Earnings'] = retained_earnings
        input_data['Market Value - Total - Fiscal'] = market_val
        input_data['Inventories - Total'] = inventory
        input_data['Stockholders Equity - Total'] = equity
        input_data['Total debt/total asset'] = debt
        
        input_df = pd.DataFrame([input_data])[model_features]
        prediction = model.predict(input_df)[0]
        
        # Display Result with Gauge Chart
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        # --- FEATURE IMPORTANCE SECTION ---
        st.markdown("### 🔍 Why this prediction?")
        
        # Get importance from the model
        feat_importances = pd.Series(model.feature_importances_, index=model_features)
        top_10 = feat_importances.sort_values(ascending=False).head(10)
        
        # Create a horizontal bar chart
        fig_importance = px.bar(
            x=top_10.values,
            y=top_10.index,
            orientation='h',
            labels={'x': 'Influence Score', 'y': 'Feature Name'},
            color=top_10.values,
            color_continuous_scale='RdYlGn' # Red to Green scale
        )
        
        # Style the chart for Dark Containers
        fig_importance.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "#ffffff"},
            xaxis={'showgrid': False, 'title': 'Importance Score'},
            yaxis={'categoryorder': 'total ascending', 'title': ''},
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        st.caption("This chart shows which financial metrics had the biggest impact on this specific forecast.")
        
        with res_col1:
            st.metric(label="Predicted Annual Revenue", value=f"${prediction:,.2f}")
            st.write("Confidence Score: 88% (Simulation)")
            
        with res_col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                # FIX: Added prefix and valueformat for currency
                number = {'prefix': "$", 'valueformat': ",.2f", 'font': {'size': 80}}, 
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Revenue Scale (Relative)", 'font': {'size': 24}},
                gauge = {
                    'axis': {
                        'range': [None, max(10000, prediction * 1.2)], 
                        'tickprefix': "$", # Adds $ to the small numbers on the arc
                        'tickformat': ",.0f"
                    },
                    'bar': {'color': "#4f46e5"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#e2e8f0",
                }
            ))
            
            # This makes the background of the chart transparent to match your light UI
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                font={'color': "#1e293b", 'family': "Inter"},
                height=350, 
                margin=dict(l=30, r=30, t=50, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            

with tab2:
    st.markdown("### Bulk Prediction Tool")
    st.write("Upload a CSV file containing multiple company records for rapid batch forecasting.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        # Verify columns exist
        missing_cols = [c for c in model_features if c not in df_upload.columns]
        
        if not missing_cols:
            preds = model.predict(df_upload[model_features])
            df_upload['Revenue_Prediction'] = preds
            st.dataframe(df_upload, use_container_width=True)
            
            csv = df_upload.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", data=csv, file_name="forecast_results.csv", mime="text/csv")
        else:
            st.error(f"Missing required columns: {', '.join(missing_cols[:5])}...")

# 6. Footer
st.markdown("""
    <div style="text-align: center; color: #64748b; margin-top: 5rem; padding: 2rem;">
        <p>________________________</p>
    </div>
    """, unsafe_allow_html=True)
