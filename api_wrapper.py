"""
Market Intelligence API
======================

Professional API wrapper for the supplement market intelligence platform.
Provides RESTful endpoints for data access, analysis, and insights.
"""

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
import uvicorn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import re

# API Models
class Product(BaseModel):
    id: Optional[int]
    name: str
    category: str
    subcategory: Optional[str]
    price: Optional[float]
    currency: str = "EUR"
    weight: Optional[str]
    ingredients: Optional[str]
    description: Optional[str]
    url: str
    scraped_at: Optional[datetime]
    price_trend: Optional[float]
    market_position: Optional[str]
    competitor_price: Optional[float]
    price_volatility: Optional[float]

class MarketAnalysis(BaseModel):
    category: str
    avg_price: float
    price_range: str
    total_products: int
    market_gaps: str
    analysis_date: datetime

class PerformanceMetrics(BaseModel):
    metric_name: str
    metric_value: float
    recorded_at: datetime

class SearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    limit: int = 50

class PricePredictionRequest(BaseModel):
    name: str
    category: str
    weight: Optional[str] = None
    ingredients: Optional[str] = None

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict[str, Any]] = None

# Initialize FastAPI app
app = FastAPI(
    title="Supplement Market Intelligence API",
    description="Professional API for supplement market intelligence and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MarketIntelligenceAPI:
    def __init__(self, db_path='supplement_market.db'):
        self.db_path = db_path
        self.ml_model = None
        self._load_ml_model()
    
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def _load_ml_model(self):
        """Load and train ML model for price prediction"""
        try:
            conn = self.get_db_connection()
            df = pd.read_sql_query('''
                SELECT name, category, price, weight, ingredients
                FROM products 
                WHERE price IS NOT NULL AND ingredients != 'N/A'
            ''', conn)
            conn.close()
            
            if len(df) > 10:
                # Feature engineering
                df['name_length'] = df['name'].str.len()
                df['ingredients_length'] = df['ingredients'].str.len()
                df['category_encoded'] = pd.Categorical(df['category']).codes
                
                # Extract weight as numeric
                df['weight_numeric'] = df['weight'].str.extract(r'(\d+(?:\.\d+)?)').astype(float)
                df['weight_numeric'].fillna(df['weight_numeric'].mean(), inplace=True)
                
                X = df[['name_length', 'ingredients_length', 'category_encoded', 'weight_numeric']]
                y = df['price']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                self.ml_model = LinearRegression()
                self.ml_model.fit(X_train, y_train)
                
                # Store model metrics
                y_pred = self.ml_model.predict(X_test)
                self.model_metrics = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'r2': r2_score(y_test, y_pred)
                }
        except Exception as e:
            print(f"ML model loading failed: {str(e)}")
            self.ml_model = None
    
    def get_products(
        self, 
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Product]:
        """Get products with optional filtering"""
        conn = self.get_db_connection()
        
        query = "SELECT * FROM products WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if min_price is not None:
            query += " AND price >= ?"
            params.append(min_price)
        
        if max_price is not None:
            query += " AND price <= ?"
            params.append(max_price)
        
        query += " ORDER BY scraped_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        products = []
        for _, row in df.iterrows():
            product_dict = row.to_dict()
            # Convert datetime strings to datetime objects
            if 'scraped_at' in product_dict and product_dict['scraped_at']:
                product_dict['scraped_at'] = pd.to_datetime(product_dict['scraped_at'])
            
            products.append(Product(**product_dict))
        
        return products
    
    def get_product_by_id(self, product_id: int) -> Optional[Product]:
        """Get a specific product by ID"""
        conn = self.get_db_connection()
        
        df = pd.read_sql_query(
            "SELECT * FROM products WHERE id = ?",
            conn,
            params=[product_id]
        )
        conn.close()
        
        if df.empty:
            return None
        
        product_dict = df.iloc[0].to_dict()
        if 'scraped_at' in product_dict and product_dict['scraped_at']:
            product_dict['scraped_at'] = pd.to_datetime(product_dict['scraped_at'])
        
        return Product(**product_dict)
    
    def get_market_analysis(self) -> List[MarketAnalysis]:
        """Get market analysis data"""
        conn = self.get_db_connection()
        
        df = pd.read_sql_query(
            "SELECT * FROM market_analysis ORDER BY analysis_date DESC",
            conn
        )
        conn.close()
        
        analyses = []
        for _, row in df.iterrows():
            analysis_dict = row.to_dict()
            if 'analysis_date' in analysis_dict and analysis_dict['analysis_date']:
                analysis_dict['analysis_date'] = pd.to_datetime(analysis_dict['analysis_date'])
            
            analyses.append(MarketAnalysis(**analysis_dict))
        
        return analyses
    
    def get_performance_metrics(self) -> List[PerformanceMetrics]:
        """Get performance metrics"""
        conn = self.get_db_connection()
        
        df = pd.read_sql_query(
            "SELECT * FROM performance_metrics ORDER BY recorded_at DESC",
            conn
        )
        conn.close()
        
        metrics = []
        for _, row in df.iterrows():
            metric_dict = row.to_dict()
            if 'recorded_at' in metric_dict and metric_dict['recorded_at']:
                metric_dict['recorded_at'] = pd.to_datetime(metric_dict['recorded_at'])
            
            metrics.append(PerformanceMetrics(**metric_dict))
        
        return metrics
    
    def get_category_statistics(self) -> Dict[str, Any]:
        """Get comprehensive category statistics"""
        conn = self.get_db_connection()
        
        # Category counts
        category_counts = pd.read_sql_query(
            "SELECT category, COUNT(*) as count FROM products GROUP BY category",
            conn
        )
        
        # Price statistics by category
        price_stats = pd.read_sql_query(
            """
            SELECT 
                category,
                COUNT(*) as total_products,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price
            FROM products 
            WHERE price IS NOT NULL
            GROUP BY category
            """,
            conn
        )
        
        # Calculate standard deviation in pandas
        if not price_stats.empty:
            std_devs = []
            for _, row in price_stats.iterrows():
                cat_data = df[df['category'] == row['category']]['price']
                std_devs.append(cat_data.std())
            price_stats['price_std'] = std_devs
        
        # Market position distribution
        market_positions = pd.read_sql_query(
            "SELECT category, market_position, COUNT(*) as count FROM products GROUP BY category, market_position",
            conn
        )
        
        conn.close()
        
        return {
            "category_counts": category_counts.to_dict('records'),
            "price_statistics": price_stats.to_dict('records'),
            "market_positions": market_positions.to_dict('records')
        }
    
    def get_price_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get price trends over time"""
        conn = self.get_db_connection()
        
        # Get price history for the last N days
        cutoff_date = datetime.now() - timedelta(days=days)
        
        price_history = pd.read_sql_query(
            """
            SELECT 
                p.category,
                ph.price,
                ph.recorded_at
            FROM price_history ph
            JOIN products p ON ph.product_id = p.id
            WHERE ph.recorded_at >= ?
            ORDER BY ph.recorded_at
            """,
            conn,
            params=[cutoff_date.isoformat()]
        )
        
        conn.close()
        
        if price_history.empty:
            return {"trends": [], "message": "No price history data available"}
        
        # Calculate daily averages by category
        price_history['recorded_at'] = pd.to_datetime(price_history['recorded_at'])
        price_history['date'] = price_history['recorded_at'].dt.date
        
        daily_avg = price_history.groupby(['date', 'category'])['price'].mean().reset_index()
        
        # Calculate trends
        trends = []
        for category in daily_avg['category'].unique():
            cat_data = daily_avg[daily_avg['category'] == category]
            if len(cat_data) > 1:
                # Simple trend calculation
                x = np.arange(len(cat_data))
                y = cat_data['price'].values
                slope = np.polyfit(x, y, 1)[0]
                
                trends.append({
                    "category": category,
                    "trend_slope": slope,
                    "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                    "data_points": len(cat_data),
                    "current_avg_price": cat_data['price'].iloc[-1]
                })
        
        return {"trends": trends, "daily_averages": daily_avg.to_dict('records')}
    
    def search_products(self, search_request: SearchRequest) -> List[Product]:
        """Advanced product search"""
        conn = self.get_db_connection()
        
        query = """
        SELECT * FROM products 
        WHERE (name LIKE ? OR description LIKE ? OR ingredients LIKE ?)
        """
        params = [f"%{search_request.query}%"] * 3
        
        if search_request.category:
            query += " AND category = ?"
            params.append(search_request.category)
        
        if search_request.min_price is not None:
            query += " AND price >= ?"
            params.append(search_request.min_price)
        
        if search_request.max_price is not None:
            query += " AND price <= ?"
            params.append(search_request.max_price)
        
        query += " ORDER BY scraped_at DESC LIMIT ?"
        params.append(search_request.limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        products = []
        for _, row in df.iterrows():
            product_dict = row.to_dict()
            if 'scraped_at' in product_dict and product_dict['scraped_at']:
                product_dict['scraped_at'] = pd.to_datetime(product_dict['scraped_at'])
            
            products.append(Product(**product_dict))
        
        return products
    
    def get_market_gaps(self) -> Dict[str, Any]:
        """Identify market gaps and opportunities"""
        conn = self.get_db_connection()
        
        df = pd.read_sql_query('''
            SELECT category, price, name
            FROM products 
            WHERE price IS NOT NULL
        ''', conn)
        conn.close()
        
        if df.empty:
            return {"gaps": [], "message": "No data available"}
        
        gaps = []
        
        # Analyze each category
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            
            if len(cat_data) > 5:
                # Price distribution analysis
                prices = cat_data['price'].values
                mean_price = np.mean(prices)
                std_price = np.std(prices)
                
                # Find price ranges with few products
                price_bins = np.linspace(prices.min(), prices.max(), 10)
                hist, _ = np.histogram(prices, bins=price_bins)
                
                # Identify gaps (bins with very few products)
                gap_threshold = np.mean(hist) * 0.3
                gap_bins = np.where(hist < gap_threshold)[0]
                
                for bin_idx in gap_bins:
                    if bin_idx < len(price_bins) - 1:
                        gap_range = (price_bins[bin_idx], price_bins[bin_idx + 1])
                        gaps.append({
                            "category": category,
                            "price_range": f"€{gap_range[0]:.2f} - €{gap_range[1]:.2f}",
                            "gap_size": gap_threshold - hist[bin_idx],
                            "opportunity": "Low competition in this price range"
                        })
        
        return {"gaps": gaps, "total_gaps": len(gaps)}
    
    def predict_price(self, request: PricePredictionRequest) -> Dict[str, Any]:
        """Predict price for a new product"""
        if not self.ml_model:
            raise HTTPException(status_code=503, detail="ML model not available")
        
        try:
            # Prepare features
            name_length = len(request.name)
            ingredients_length = len(request.ingredients) if request.ingredients else 0
            
            # Encode category
            categories = ['Protein', 'Nutraceuticals', 'Bars', 'Apparel']
            category_encoded = categories.index(request.category) if request.category in categories else 0
            
            # Extract weight
            weight_numeric = 0
            if request.weight:
                weight_match = re.search(r'(\d+(?:\.\d+)?)', request.weight)
                if weight_match:
                    weight_numeric = float(weight_match.group(1))
            
            # Make prediction
            features = np.array([[name_length, ingredients_length, category_encoded, weight_numeric]])
            predicted_price = self.ml_model.predict(features)[0]
            
            return {
                "predicted_price": round(predicted_price, 2),
                "confidence": "medium",  # Could be enhanced with prediction intervals
                "model_metrics": self.model_metrics,
                "features_used": {
                    "name_length": name_length,
                    "ingredients_length": ingredients_length,
                    "category": request.category,
                    "weight": weight_numeric
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def get_ingredients_analysis(self) -> Dict[str, Any]:
        """Analyze most common ingredients"""
        conn = self.get_db_connection()
        
        df = pd.read_sql_query('''
            SELECT ingredients, category
            FROM products 
            WHERE ingredients != 'N/A' AND ingredients IS NOT NULL
        ''', conn)
        conn.close()
        
        if df.empty:
            return {"ingredients": [], "message": "No ingredient data available"}
        
        # Extract ingredients from text
        all_ingredients = []
        for ingredients_text in df['ingredients']:
            if ingredients_text:
                # Simple ingredient extraction (split by common separators)
                ingredients = re.split(r'[,;()]', ingredients_text)
                for ingredient in ingredients:
                    ingredient = ingredient.strip().lower()
                    if len(ingredient) > 2 and ingredient not in ['', 'and', 'or', 'with']:
                        all_ingredients.append(ingredient)
        
        # Count frequencies
        ingredient_counts = pd.Series(all_ingredients).value_counts().head(20)
        
        return {
            "top_ingredients": ingredient_counts.to_dict(),
            "total_unique_ingredients": len(set(all_ingredients)),
            "total_products_with_ingredients": len(df)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        conn = self.get_db_connection()
        
        # Get all products data for calculations
        df = pd.read_sql_query("SELECT * FROM products", conn)
        
        # Category counts
        category_counts = pd.read_sql_query(
            "SELECT category, COUNT(*) as count FROM products GROUP BY category",
            conn
        )
        
        # Price statistics by category
        price_stats = pd.read_sql_query(
            """
            SELECT 
                category,
                COUNT(*) as total_products,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price
            FROM products 
            WHERE price IS NOT NULL
            GROUP BY category
            """,
            conn
        )
        
        # Calculate standard deviation in pandas
        if not price_stats.empty:
            std_devs = []
            for _, row in price_stats.iterrows():
                cat_data = df[df['category'] == row['category']]['price']
                std_devs.append(cat_data.std())
            price_stats['price_std'] = std_devs
        
        # Market position distribution
        market_positions = pd.read_sql_query(
            "SELECT category, market_position, COUNT(*) as count FROM products GROUP BY category, market_position",
            conn
        )
        
        # Data quality metrics
        data_quality = pd.read_sql_query(
            """
            SELECT 
                COUNT(*) as total_products,
                SUM(CASE WHEN price IS NOT NULL THEN 1 ELSE 0 END) as products_with_price,
                SUM(CASE WHEN ingredients != 'N/A' THEN 1 ELSE 0 END) as products_with_ingredients,
                SUM(CASE WHEN weight != 'N/A' THEN 1 ELSE 0 END) as products_with_weight
            FROM products
            """,
            conn
        )
        
        conn.close()
        
        return {
            "category_counts": category_counts.to_dict('records'),
            "price_statistics": price_stats.to_dict('records'),
            "market_positions": market_positions.to_dict('records'),
            "data_quality": data_quality.to_dict('records')[0]
        }

# Initialize API instance
api = MarketIntelligenceAPI()

# API Endpoints

@app.get("/", response_model=APIResponse)
async def root():
    """Root endpoint with API information"""
    return APIResponse(
        success=True,
        message="Supplement Market Intelligence API",
        data={
            "version": "1.0.0",
            "description": "Professional API for supplement market intelligence",
            "endpoints": {
                "products": "/api/products",
                "market_analysis": "/api/market-analysis",
                "performance": "/api/performance",
                "statistics": "/api/statistics",
                "trends": "/api/trends",
                "search": "/api/search",
                "market-gaps": "/api/market-gaps",
                "predict-price": "/api/predict-price",
                "ingredients": "/api/ingredients",
                "health": "/api/health"
            }
        }
    )

@app.get("/api/products", response_model=APIResponse)
async def get_products(
    category: Optional[str] = Query(None, description="Filter by category"),
    min_price: Optional[float] = Query(None, description="Minimum price filter"),
    max_price: Optional[float] = Query(None, description="Maximum price filter"),
    limit: int = Query(100, description="Number of products to return"),
    offset: int = Query(0, description="Number of products to skip")
):
    """Get products with optional filtering"""
    try:
        products = api.get_products(category, min_price, max_price, limit, offset)
        return APIResponse(
            success=True,
            message=f"Retrieved {len(products)} products",
            data=[product.dict() for product in products],
            metadata={"total": len(products), "category": category}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/products/{product_id}", response_model=APIResponse)
async def get_product(product_id: int):
    """Get a specific product by ID"""
    try:
        product = api.get_product_by_id(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        return APIResponse(
            success=True,
            message="Product retrieved successfully",
            data=product.dict()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-analysis", response_model=APIResponse)
async def get_market_analysis():
    """Get market analysis data"""
    try:
        analysis = api.get_market_analysis()
        return APIResponse(
            success=True,
            message=f"Retrieved {len(analysis)} market analysis records",
            data=[record.dict() for record in analysis]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/performance", response_model=APIResponse)
async def get_performance_metrics():
    """Get performance metrics"""
    try:
        metrics = api.get_performance_metrics()
        return APIResponse(
            success=True,
            message=f"Retrieved {len(metrics)} performance metrics",
            data=[metric.dict() for metric in metrics]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/statistics", response_model=APIResponse)
async def get_statistics():
    """Get comprehensive statistics"""
    try:
        stats = api.get_statistics()
        return APIResponse(
            success=True,
            message="Statistics retrieved successfully",
            data=stats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trends", response_model=APIResponse)
async def get_price_trends(days: int = Query(30, description="Number of days to analyze")):
    """Get price trends over time"""
    try:
        trends = api.get_price_trends(days)
        return APIResponse(
            success=True,
            message=f"Price trends analyzed for last {days} days",
            data=trends
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search", response_model=APIResponse)
async def search_products(search_request: SearchRequest):
    """Advanced product search"""
    try:
        products = api.search_products(search_request)
        return APIResponse(
            success=True,
            message=f"Found {len(products)} products matching '{search_request.query}'",
            data=[product.dict() for product in products],
            metadata={"query": search_request.query, "filters_applied": search_request.dict()}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market-gaps", response_model=APIResponse)
async def get_market_gaps():
    """Identify market gaps and opportunities"""
    try:
        gaps = api.get_market_gaps()
        return APIResponse(
            success=True,
            message=f"Found {gaps.get('total_gaps', 0)} market gaps",
            data=gaps
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict-price", response_model=APIResponse)
async def predict_price(request: PricePredictionRequest):
    """Predict price for a new product using ML"""
    try:
        prediction = api.predict_price(request)
        return APIResponse(
            success=True,
            message="Price prediction completed",
            data=prediction
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ingredients", response_model=APIResponse)
async def get_ingredients_analysis():
    """Analyze most common ingredients"""
    try:
        analysis = api.get_ingredients_analysis()
        return APIResponse(
            success=True,
            message="Ingredient analysis completed",
            data=analysis
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health", response_model=APIResponse)
async def health_check():
    """Health check endpoint"""
    try:
        conn = api.get_db_connection()
        conn.execute("SELECT 1")
        conn.close()
        
        return APIResponse(
            success=True,
            message="API is healthy",
            data={
                "status": "healthy",
                "database": "connected",
                "ml_model": "loaded" if api.ml_model else "not_available",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        return APIResponse(
            success=False,
            message="API health check failed",
            data={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 