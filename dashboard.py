"""
Marketing Mix Modeling (MMM) Dashboard

Main presentation dashboard for MMM pipeline results.
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
import json
from datetime import datetime, timedelta

# Import MMM modules
import sys
sys.path.append(str(Path(__file__).parent))

from src.modeling import MMMModel
from src.optimization import BudgetOptimizer
from src.monitoring import ModelMonitor
from src.utils.config import load_config

# Page configuration
st.set_page_config(
    page_title="MMM Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(data_path: str):
    """Load and cache data."""
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)


@st.cache_resource
def load_model(model_path: str):
    """Load and cache model."""
    return joblib.load(model_path)


def main():
    """Main dashboard application."""
    
    # Title
    st.markdown('<h1 class="main-header">📊 Marketing Mix Modeling Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Data selection
        data_path = st.text_input(
            "Data Path",
            value="data/raw/sample_data.csv",
            help="Path to input data CSV file"
        )
        
        # Model selection
        model_path = st.text_input(
            "Model Path",
            value="",
            help="Path to trained model (leave empty to train new)"
        )
        
        # Budget for optimization
        total_budget = st.number_input(
            "Total Budget ($)",
            min_value=10000,
            max_value=10000000,
            value=1000000,
            step=10000,
            help="Total media budget to optimize"
        )
        
        st.markdown("---")
        st.markdown("### 📈 Quick Actions")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        **MMM Dashboard** showcases:
        - Model performance metrics
        - Channel ROI analysis
        - Budget optimization
        - Response curves
        - Scenario planning
        """)
    
    # Main content
    try:
        # Load data
        if Path(data_path).exists():
            df = load_data(data_path)
            
            # Tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Overview",
                "💰 Channel ROI",
                "🎯 Budget Optimization",
                "📈 Response Curves",
                "🔍 Model Performance"
            ])
            
            with tab1:
                show_overview(df)
            
            with tab2:
                show_channel_roi(df, model_path)
            
            with tab3:
                show_budget_optimization(df, model_path, total_budget)
            
            with tab4:
                show_response_curves(df, model_path)
            
            with tab5:
                show_model_performance(df, model_path)
                
        else:
            st.error(f"❌ Data file not found: {data_path}")
            st.info("💡 Generate sample data first: `python scripts/generate_sample_data.py`")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)


def show_overview(df):
    """Show overview dashboard."""
    st.header("📊 Data Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Revenue",
            f"${df['revenue'].sum():,.0f}",
            delta=f"${df['revenue'].mean():,.0f} avg/day"
        )
    
    with col2:
        spend_cols = [col for col in df.columns if col.startswith('spend_')]
        total_spend = df[spend_cols].sum().sum()
        st.metric(
            "Total Media Spend",
            f"${total_spend:,.0f}",
            delta=f"${df[spend_cols].sum(axis=1).mean():,.0f} avg/day"
        )
    
    with col3:
        roi = df['revenue'].sum() / total_spend if total_spend > 0 else 0
        st.metric(
            "Overall ROI",
            f"{roi:.2f}x",
            delta=f"${df['revenue'].sum() - total_spend:,.0f} profit"
        )
    
    with col4:
        st.metric(
            "Date Range",
            f"{(df['date'].max() - df['date'].min()).days} days",
            delta=f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        )
    
    st.markdown("---")
    
    # Revenue and spend over time
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Revenue Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['revenue'],
            mode='lines',
            name='Revenue',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            title="Daily Revenue Over Time",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💵 Media Spend by Channel")
        spend_cols = [col for col in df.columns if col.startswith('spend_')]
        spend_df = df[['date'] + spend_cols].set_index('date').resample('W').sum()
        
        fig = go.Figure()
        for col in spend_cols:
            channel = col.replace('spend_', '')
            fig.add_trace(go.Scatter(
                x=spend_df.index,
                y=spend_df[col],
                mode='lines',
                name=channel,
                stackgroup='one'
            ))
        fig.update_layout(
            title="Weekly Media Spend (Stacked)",
            xaxis_title="Date",
            yaxis_title="Spend ($)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel spend distribution
    st.subheader("📊 Channel Spend Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        total_by_channel = df[spend_cols].sum().sort_values(ascending=False)
        fig = px.bar(
            x=total_by_channel.values,
            y=[col.replace('spend_', '') for col in total_by_channel.index],
            orientation='h',
            title="Total Spend by Channel",
            labels={'x': 'Total Spend ($)', 'y': 'Channel'},
            color=total_by_channel.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Correlation heatmap
        corr_cols = ['revenue'] + spend_cols[:6]
        corr_matrix = df[corr_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Revenue vs Spend Correlation",
            color_continuous_scale='RdBu',
            labels=dict(x="Variable", y="Variable", color="Correlation")
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def show_channel_roi(df, model_path):
    """Show channel ROI analysis."""
    st.header("💰 Channel ROI Analysis")
    
    if not model_path or not Path(model_path).exists():
        st.warning("⚠️ No trained model found. Training model on the fly...")
        with st.spinner("Training model..."):
            model = train_model_quick(df)
    else:
        model = load_model(model_path)
    
    if model and model.is_trained:
        # Get ROI
        roi_df = model.get_channel_roi(df)
        
        # Display ROI metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Channel ROI Ranking")
            fig = px.bar(
                roi_df.sort_values('roi', ascending=True),
                x='roi',
                y='channel',
                orientation='h',
                title="ROI by Channel",
                labels={'roi': 'ROI (Revenue/Spend)', 'y': 'Channel'},
                color='roi',
                color_continuous_scale='Greens'
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("💵 Channel Contribution")
            contribution_df = pd.DataFrame({
                'channel': roi_df['channel'],
                'contribution': roi_df['contribution'],
                'spend': [df[f'spend_{ch}'].sum() for ch in roi_df['channel']]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=contribution_df['channel'],
                y=contribution_df['contribution'],
                name='Revenue Contribution',
                marker_color='#1f77b4'
            ))
            fig.add_trace(go.Bar(
                x=contribution_df['channel'],
                y=contribution_df['spend'],
                name='Total Spend',
                marker_color='#ff7f0e'
            ))
            fig.update_layout(
                title="Revenue Contribution vs Spend",
                xaxis_title="Channel",
                yaxis_title="Amount ($)",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # ROI table
        st.subheader("📋 Detailed ROI Metrics")
        display_df = roi_df.copy()
        display_df['roi'] = display_df['roi'].apply(lambda x: f"{x:.2f}x")
        display_df['contribution'] = display_df['contribution'].apply(lambda x: f"${x:,.0f}")
        display_df['efficiency'] = display_df['efficiency'].apply(lambda x: f"{x:.2f}")
        st.dataframe(display_df, use_container_width=True)
    else:
        st.error("❌ Model not trained. Please train a model first.")


def show_budget_optimization(df, model_path, total_budget):
    """Show budget optimization results."""
    st.header("🎯 Budget Optimization")
    
    if not model_path or not Path(model_path).exists():
        st.warning("⚠️ No trained model found. Training model on the fly...")
        with st.spinner("Training model..."):
            model = train_model_quick(df)
    else:
        model = load_model(model_path)
    
    if model and model.is_trained:
        # Optimize budget
        st.subheader(f"💰 Optimizing ${total_budget:,.0f} Budget")
        
        with st.spinner("Optimizing budget allocation..."):
            optimizer = BudgetOptimizer(model, method='scipy')
            base_data = df.tail(1)
            
            optimal_budget = optimizer.optimize(
                total_budget=total_budget,
                channels=model.media_channels,
                base_data=base_data
            )
            
            # Compare scenarios
            scenarios_df = optimizer.compare_scenarios(
                total_budget=total_budget,
                channels=model.media_channels,
                base_data=base_data
            )
        
        # Display optimal allocation
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Optimal Budget Allocation")
            opt_df = pd.DataFrame([
                {'Channel': k, 'Budget': v, 'Percentage': v/total_budget*100}
                for k, v in optimal_budget.items()
            ])
            
            fig = px.pie(
                opt_df,
                values='Budget',
                names='Channel',
                title="Optimal Budget Distribution",
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Budget table
            opt_df['Budget'] = opt_df['Budget'].apply(lambda x: f"${x:,.0f}")
            opt_df['Percentage'] = opt_df['Percentage'].apply(lambda x: f"{x:.1f}%")
            st.dataframe(opt_df, use_container_width=True)
        
        with col2:
            st.subheader("📈 Scenario Comparison")
            st.dataframe(scenarios_df, use_container_width=True)
            
            # Scenario comparison chart
            fig = px.bar(
                scenarios_df,
                x='scenario',
                y='total_revenue',
                title="Revenue by Scenario",
                labels={'total_revenue': 'Total Revenue ($)', 'scenario': 'Scenario'},
                color='total_revenue',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Expected results
        st.subheader("🎯 Expected Results")
        predicted_revenue = scenarios_df[scenarios_df['scenario'] == 'Optimal']['total_revenue'].values[0]
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted Revenue", f"${predicted_revenue:,.0f}")
        with col2:
            st.metric("Total Budget", f"${total_budget:,.0f}")
        with col3:
            st.metric("Expected ROI", f"{(predicted_revenue/total_budget):.2f}x")
    else:
        st.error("❌ Model not trained. Please train a model first.")


def show_response_curves(df, model_path):
    """Show media response curves."""
    st.header("📈 Media Response Curves")
    
    if not model_path or not Path(model_path).exists():
        st.warning("⚠️ No trained model found. Training model on the fly...")
        with st.spinner("Training model..."):
            model = train_model_quick(df)
    else:
        model = load_model(model_path)
    
    if model and model.is_trained:
        # Generate response curves
        st.subheader("🎯 Diminishing Returns Analysis")
        
        channels = st.multiselect(
            "Select Channels",
            options=model.media_channels,
            default=model.media_channels[:3]
        )
        
        if channels:
            fig = go.Figure()
            
            for channel in channels:
                # Generate spend range
                max_spend = df[f'spend_{channel}'].max() * 2
                spend_range = np.linspace(0, max_spend, 100)
                
                # Apply saturation
                from src.feature_engineering import SaturationTransformer
                sat_params = model.saturation_params.get(channel, {'method': 'hill', 'alpha': 0.5, 'gamma': 0.5})
                sat_transformer = SaturationTransformer(**sat_params)
                saturated = sat_transformer.transform(spend_range)
                
                # Estimate revenue using model coefficients
                base_revenue = model.model.intercept_ if model.model else df['revenue'].mean()
                
                # Find coefficient for this channel
                channel_coef = 10  # Default
                for feat_name, coef in model.coefficients.items():
                    if channel.lower() in feat_name.lower():
                        channel_coef = coef
                        break
                
                # Scale saturated values (they're normalized, so we need to scale back)
                # Simplified: assume spend_range is in original scale
                revenue = base_revenue + channel_coef * saturated * (max_spend / 100)
                
                fig.add_trace(go.Scatter(
                    x=spend_range,
                    y=revenue,
                    mode='lines',
                    name=channel,
                    line=dict(width=3)
                ))
            
            fig.update_layout(
                title="Media Response Curves (Diminishing Returns)",
                xaxis_title="Media Spend ($)",
                yaxis_title="Predicted Revenue ($)",
                height=500,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 These curves show diminishing returns - each additional dollar of spend generates less incremental revenue.")
    else:
        st.error("❌ Model not trained. Please train a model first.")


def show_model_performance(df, model_path):
    """Show model performance metrics."""
    st.header("🔍 Model Performance")
    
    if not model_path or not Path(model_path).exists():
        st.warning("⚠️ No trained model found. Training model on the fly...")
        with st.spinner("Training model..."):
            model = train_model_quick(df)
    else:
        model = load_model(model_path)
    
    if model and model.is_trained:
        # Evaluate model
        metrics = model.evaluate(df)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{metrics.get('r2_score', 0):.4f}")
        with col2:
            st.metric("MAE", f"${metrics.get('mae', 0):,.0f}")
        with col3:
            st.metric("RMSE", f"${metrics.get('rmse', 0):,.0f}")
        with col4:
            st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
        
        # Actual vs Predicted
        st.subheader("📊 Actual vs Predicted Revenue")
        predictions = model.predict(df)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['revenue'],
            mode='lines',
            name='Actual',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=predictions,
            mode='lines',
            name='Predicted',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        fig.update_layout(
            title="Revenue: Actual vs Predicted",
            xaxis_title="Date",
            yaxis_title="Revenue ($)",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals plot
        st.subheader("📉 Residuals Analysis")
        residuals = df['revenue'] - predictions
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=predictions,
                y=residuals,
                mode='markers',
                marker=dict(color='#1f77b4', size=5, opacity=0.5)
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(
                title="Residuals vs Predicted",
                xaxis_title="Predicted Revenue ($)",
                yaxis_title="Residuals ($)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=residuals,
                nbinsx=30,
                marker_color='#1f77b4'
            ))
            fig.update_layout(
                title="Residuals Distribution",
                xaxis_title="Residuals ($)",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ Model not trained. Please train a model first.")


def train_model_quick(df):
    """Quickly train a model for demo purposes."""
    try:
        media_channels = [col.replace('spend_', '') for col in df.columns if col.startswith('spend_')]
        external_factors = [col for col in df.columns if col in ['gdp_growth', 'unemployment', 'inflation', 'holiday_flag']]
        
        model = MMMModel(
            media_channels=media_channels,
            external_factors=external_factors
        )
        model.train(df)
        return model
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None


if __name__ == "__main__":
    main()

