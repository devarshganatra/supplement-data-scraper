"""
Market Intelligence Dashboard
============================

Interactive dashboard for supplement market intelligence platform.
Provides real-time insights, price analysis, and market trends.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import numpy as np
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(
    page_title="Supplement Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-metric {
        border-left-color: #28a745;
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
    }
    .warning-metric {
        border-left-color: #ffc107;
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    }
    .danger-metric {
        border-left-color: #dc3545;
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .download-button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.3rem;
        cursor: pointer;
        margin: 0.5rem;
    }
    .landing-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MarketIntelligenceDashboard:
    def __init__(self, db_path='supplement_market.db'):
        self.db_path = db_path
        
    def load_data(self):
        """Load data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load products data
            products_df = pd.read_sql_query('''
                SELECT * FROM products ORDER BY scraped_at DESC
            ''', conn)
            
            # Load market analysis
            market_analysis_df = pd.read_sql_query('''
                SELECT * FROM market_analysis ORDER BY analysis_date DESC
            ''', conn)
            
            # Load performance metrics
            performance_df = pd.read_sql_query('''
                SELECT * FROM performance_metrics ORDER BY recorded_at DESC
            ''', conn)
            
            conn.close()
            
            return products_df, market_analysis_df, performance_df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def create_dashboard(self):
        """Create the main dashboard"""
        # Header with branding
        st.markdown('<h1 class="main-header">📊 Supplement Market Intelligence Platform</h1>', unsafe_allow_html=True)
        st.markdown("### Built for market research to analyze supplement pricing trends")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["🏠 Dashboard", "📊 Data Analysis", "🔍 Market Insights", "🤖 ML Insights", "📈 Reports", "⚙️ Settings"]
        )
        
        # Load data
        products_df, market_analysis_df, performance_df = self.load_data()
        
        if page == "🏠 Dashboard":
            self.show_dashboard_home(products_df, performance_df)
        elif page == "📊 Data Analysis":
            self.show_data_analysis(products_df)
        elif page == "🔍 Market Insights":
            self.show_market_insights(products_df)
        elif page == "🤖 ML Insights":
            self.show_ml_insights(products_df)
        elif page == "📈 Reports":
            self.show_reports(products_df, market_analysis_df)
        elif page == "⚙️ Settings":
            self.show_settings()
    
    def show_dashboard_home(self, products_df, performance_df):
        """Show the main dashboard home page"""
        # Landing section
        st.markdown("""
        <div class="landing-section">
            <h2>🎯 Welcome to Market Intelligence</h2>
            <p>Get real-time insights into supplement market trends, pricing analysis, and competitive intelligence.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_products = len(products_df)
            st.markdown(f"""
            <div class="metric-card success-metric">
                <h3>📦 Total Products</h3>
                <h2>{total_products:,}</h2>
                <p>Successfully scraped</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'price' in products_df.columns and not products_df.empty:
                avg_price = products_df['price'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💰 Average Price</h3>
                    <h2>€{avg_price:.2f}</h2>
                    <p>Market average</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            categories = products_df['category'].nunique() if not products_df.empty else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>📂 Categories</h3>
                <h2>{categories}</h2>
                <p>Product categories</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Calculate accuracy from performance metrics
            if not performance_df.empty:
                accuracy_row = performance_df[performance_df['metric_name'] == 'scraping_accuracy']
                if not accuracy_row.empty:
                    accuracy = accuracy_row.iloc[0]['metric_value']
                    st.markdown(f"""
                    <div class="metric-card success-metric">
                        <h3>🎯 Accuracy</h3>
                        <h2>{accuracy:.1f}%</h2>
                        <p>Scraping success rate</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Data quality section
        st.subheader("🔍 Data Quality Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            if not products_df.empty:
                # Data completeness
                missing_data = products_df.isnull().sum()
                completeness = ((len(products_df) - missing_data) / len(products_df) * 100).round(2)
                
                fig = go.Figure(data=[
                    go.Bar(x=completeness.index, y=completeness.values, 
                           marker_color=['#28a745' if x > 90 else '#ffc107' if x > 70 else '#dc3545' for x in completeness.values])
                ])
                fig.update_layout(title="Data Completeness by Field (%)", height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if not products_df.empty and 'price' in products_df.columns:
                # Price distribution
                fig = px.histogram(products_df, x='price', nbins=20, 
                                 title="Price Distribution",
                                 labels={'price': 'Price (EUR)', 'count': 'Number of Products'})
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        if not products_df.empty and 'scraped_at' in products_df.columns:
            products_df['scraped_at'] = pd.to_datetime(products_df['scraped_at'])
            recent_activity = products_df.groupby(products_df['scraped_at'].dt.date).size().tail(7)
            
            fig = px.line(x=recent_activity.index, y=recent_activity.values,
                         title="Products Scraped (Last 7 Days)",
                         labels={'x': 'Date', 'y': 'Products Scraped'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def show_data_analysis(self, products_df):
        """Show detailed data analysis"""
        st.subheader("📊 Data Analysis")
        
        if products_df.empty:
            st.warning("No data available for analysis.")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            categories = ['All'] + list(products_df['category'].unique())
            selected_category = st.selectbox("Category", categories)
        
        with col2:
            if 'price' in products_df.columns:
                min_price = float(products_df['price'].min()) if not products_df['price'].isna().all() else 0
                max_price = float(products_df['price'].max()) if not products_df['price'].isna().all() else 100
                price_range = st.slider("Price Range (EUR)", min_price, max_price, (min_price, max_price))
        
        with col3:
            search_term = st.text_input("Search Products", "")
        
        # Filter data
        filtered_df = products_df.copy()
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        if 'price' in filtered_df.columns:
            filtered_df = filtered_df[(filtered_df['price'] >= price_range[0]) & (filtered_df['price'] <= price_range[1])]
        if search_term:
            filtered_df = filtered_df[filtered_df['name'].str.contains(search_term, case=False, na=False)]
        
        # Analysis charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Category breakdown
            category_counts = filtered_df['category'].value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index, 
                        title="Products by Category")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Price by category
            if 'price' in filtered_df.columns:
                fig = px.box(filtered_df, x='category', y='price', 
                           title="Price Distribution by Category")
                st.plotly_chart(fig, use_container_width=True)
        
        # Data table with download
        st.subheader("📋 Product Data")
        display_columns = ['name', 'category', 'price', 'weight', 'market_position']
        available_columns = [col for col in display_columns if col in filtered_df.columns]
        st.dataframe(filtered_df[available_columns].head(20))
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"supplement_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, sheet_name='Products', index=False)
            excel_data = excel_buffer.getvalue()
            st.download_button(
                label="📥 Download Excel",
                data=excel_data,
                file_name=f"supplement_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    def show_market_insights(self, products_df):
        """Show market insights and trends"""
        st.subheader("🔍 Market Insights")
        
        if products_df.empty:
            st.warning("No data available for market insights.")
            return
        
        # Market position analysis
        col1, col2 = st.columns(2)
        
        with col1:
            if 'market_position' in products_df.columns:
                position_counts = products_df['market_position'].value_counts()
                fig = px.bar(x=position_counts.index, y=position_counts.values,
                           title="Products by Market Position",
                           labels={'x': 'Market Position', 'y': 'Number of Products'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'price' in products_df.columns and 'category' in products_df.columns:
                # Price trends by category
                category_prices = products_df.groupby('category')['price'].agg(['mean', 'min', 'max']).reset_index()
                fig = go.Figure()
                
                for _, row in category_prices.iterrows():
                    fig.add_trace(go.Bar(
                        name=row['category'],
                        x=[row['category']],
                        y=[row['mean']],
                        text=f"€{row['mean']:.2f}",
                        textposition='auto',
                    ))
                
                fig.update_layout(title="Average Price by Category", yaxis_title="Price (EUR)")
                st.plotly_chart(fig, use_container_width=True)
        
        # Market gaps analysis
        st.subheader("🎯 Market Gap Analysis")
        
        if 'price' in products_df.columns and 'category' in products_df.columns:
            # Identify price gaps
            category_stats = products_df.groupby('category')['price'].describe()
            
            # Find categories with high price variance (potential gaps)
            price_variance = category_stats['std'] / category_stats['mean']
            
            fig = px.bar(x=price_variance.index, y=price_variance.values,
                        title="Price Variance by Category (Market Gap Indicator)",
                        labels={'x': 'Category', 'y': 'Coefficient of Variation'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Market gap insights
            st.markdown("**Market Gap Insights:**")
            high_variance_categories = price_variance[price_variance > price_variance.mean()]
            if not high_variance_categories.empty:
                for category in high_variance_categories.index:
                    st.write(f"• **{category}**: High price variance indicates market gaps")
            else:
                st.write("• All categories show relatively stable pricing")
    
    def show_ml_insights(self, products_df):
        """Show ML insights and predictions"""
        st.subheader("🤖 Machine Learning Insights")
        
        if products_df.empty:
            st.warning("No data available for ML analysis.")
            return
        
        # Simple price prediction model
        st.markdown("### 💰 Price Prediction Model")
        
        if 'price' in products_df.columns and 'category' in products_df.columns:
            # Simple feature engineering
            ml_df = products_df.copy()
            ml_df['name_length'] = ml_df['name'].str.len()
            ml_df['category_encoded'] = pd.Categorical(ml_df['category']).codes
            
            # Remove rows with missing price
            ml_df = ml_df.dropna(subset=['price'])
            
            if len(ml_df) > 10:
                # Simple linear regression for demo
                from sklearn.linear_model import LinearRegression
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_absolute_error, r2_score
                
                X = ml_df[['name_length', 'category_encoded']]
                y = ml_df['price']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Display model performance
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model Accuracy (R²)", f"{r2:.3f}")
                
                with col2:
                    st.metric("Mean Absolute Error", f"€{mae:.2f}")
                
                with col3:
                    st.metric("Training Samples", len(X_train))
                
                # Predicted vs Actual
                fig = px.scatter(x=y_test, y=y_pred, 
                               title="Predicted vs Actual Prices",
                               labels={'x': 'Actual Price (EUR)', 'y': 'Predicted Price (EUR)'})
                fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                       y=[y_test.min(), y_test.max()], 
                                       mode='lines', name='Perfect Prediction'))
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'Feature': ['Name Length', 'Category'],
                    'Importance': [abs(model.coef_[0]), abs(model.coef_[1])]
                })
                
                fig = px.bar(feature_importance, x='Feature', y='Importance',
                           title="Feature Importance")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient data for ML model training.")
    
    def show_reports(self, products_df, market_analysis_df):
        """Show reports and analytics"""
        st.subheader("📈 Reports & Analytics")
        
        if products_df.empty:
            st.warning("No data available for reports.")
            return
        
        # Generate comprehensive report
        st.markdown("### 📊 Market Intelligence Report")
        
        # Executive summary
        st.markdown("**Executive Summary:**")
        total_products = len(products_df)
        avg_price = products_df['price'].mean() if 'price' in products_df.columns else 0
        categories = products_df['category'].nunique()
        
        st.write(f"• Total products analyzed: {total_products}")
        st.write(f"• Average market price: €{avg_price:.2f}")
        st.write(f"• Product categories: {categories}")
        
        # Top products by price
        if 'price' in products_df.columns:
            st.markdown("**Top 10 Most Expensive Products:**")
            top_products = products_df.nlargest(10, 'price')[['name', 'category', 'price']]
            st.dataframe(top_products, use_container_width=True)
        
        # Category analysis
        if 'category' in products_df.columns:
            st.markdown("**Category Analysis:**")
            category_analysis = products_df.groupby('category').agg({
                'price': ['count', 'mean', 'std']
            }).round(2)
            st.dataframe(category_analysis, use_container_width=True)
    
    def show_settings(self):
        """Show settings and configuration"""
        st.subheader("⚙️ Settings & Configuration")
        
        st.markdown("### Database Configuration")
        st.info(f"Current database: {self.db_path}")
        
        st.markdown("### Data Refresh")
        if st.button("🔄 Refresh Data"):
            st.success("Data refresh initiated!")
        
        st.markdown("### Export Configuration")
        st.checkbox("Include ML predictions in exports", value=True)
        st.checkbox("Include data quality metrics", value=True)
        
        st.markdown("### System Information")
        st.write(f"• Platform: Market Intelligence Dashboard")
        st.write(f"• Version: 1.0.0")
        st.write(f"• Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main dashboard function"""
    dashboard = MarketIntelligenceDashboard()
    dashboard.create_dashboard()

if __name__ == "__main__":
    main() 