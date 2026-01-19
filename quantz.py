"""
Complete Quantitative Beta Analysis System
Download and run: python quantz_system.py

Installation:
pip install fastapi uvicorn pandas numpy yfinance pydantic sqlalchemy plotly

Usage:
python quantz_system.py
Then visit: http://localhost:8000/docs for API documentation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import sqlite3
import yfinance as yf
from contextlib import contextmanager
import uvicorn
import json

# ============================================================================
# ENUMERATIONS
# ============================================================================

class PricebarTimeFrame(str, Enum):
    DAILY = "Daily"
    HOURLY = "Hourly"
    MIN_15 = "15Min"
    MIN_5 = "5Min"

class RollingWindow(str, Enum):
    ONE_MONTH = "1Month"
    TWO_MONTH = "2Month"
    THREE_MONTH = "3Month"

class LookbackPeriod(str, Enum):
    THREE_MONTH = "3Month"
    TWO_YEAR = "2Year"
    FIVE_YEAR = "5Year"

# ============================================================================
# DATABASE MODELS
# ============================================================================

class BetaRecord(BaseModel):
    timestamp: datetime
    security_id: int
    pricebar_timeframe: PricebarTimeFrame
    rolling_window: RollingWindow
    reference_index: str
    lookback_period: LookbackPeriod
    value: float

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    def __init__(self):
        self.init_databases()
    
    @contextmanager
    def get_connection(self, db_name: str):
        if db_name == "price":
            conn = sqlite3.connect("price.db")
        elif db_name == "quantz":
            conn = sqlite3.connect("quantz.db")
        else:
            conn = sqlite3.connect("quantz.db")
        try:
            yield conn
        finally:
            conn.close()
    
    def init_databases(self):
        with self.get_connection("price") as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    security_id INTEGER,
                    timestamp DATETIME,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    UNIQUE(security_id, timestamp)
                )
            """)
            conn.commit()
        
        with self.get_connection("quantz") as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS beta (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    security_id INTEGER,
                    pricebar_timeframe TEXT,
                    rolling_window TEXT,
                    reference_index TEXT,
                    lookback_period TEXT,
                    value REAL,
                    UNIQUE(timestamp, security_id, pricebar_timeframe, rolling_window, reference_index, lookback_period)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS security_master (
                    security_id INTEGER PRIMARY KEY,
                    symbol TEXT UNIQUE,
                    name TEXT
                )
            """)
            conn.commit()
    
    def store_ohlcv_data(self, security_id: int, df: pd.DataFrame):
        with self.get_connection("price") as conn:
            for idx, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO ohlcv 
                        (security_id, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (security_id, str(idx), float(row['Open']), float(row['High']), 
                          float(row['Low']), float(row['Close']), int(row['Volume'])))
                except Exception as e:
                    print(f"Error storing row: {e}")
            conn.commit()
    
    def store_beta(self, record: BetaRecord):
        with self.get_connection("quantz") as conn:
            conn.execute("""
                INSERT OR REPLACE INTO beta 
                (timestamp, security_id, pricebar_timeframe, rolling_window, 
                 reference_index, lookback_period, value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (str(record.timestamp), record.security_id, record.pricebar_timeframe,
                  record.rolling_window, record.reference_index, 
                  record.lookback_period, record.value))
            conn.commit()
    
    def add_security(self, security_id: int, symbol: str, name: str):
        with self.get_connection("quantz") as conn:
            conn.execute("""
                INSERT OR REPLACE INTO security_master (security_id, symbol, name)
                VALUES (?, ?, ?)
            """, (security_id, symbol, name))
            conn.commit()

# ============================================================================
# QUANTZLIB - Beta Calculation Library
# ============================================================================

class Quantzlib:
    @staticmethod
    def calculate_beta(stock_returns: pd.Series, 
                      market_returns: pd.Series,
                      rolling_window: Optional[int] = None):
        if rolling_window:
            covariance = stock_returns.rolling(window=rolling_window).cov(market_returns)
            variance = market_returns.rolling(window=rolling_window).var()
            beta = covariance / variance
            return beta.dropna()
        else:
            aligned_stock, aligned_market = stock_returns.align(market_returns, join='inner')
            covariance = np.cov(aligned_stock, aligned_market)[0, 1]
            variance = np.var(aligned_market)
            return covariance / variance if variance != 0 else 0
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        return prices.pct_change().dropna()

# ============================================================================
# PRICE SERVICE
# ============================================================================

class PriceService:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def fetch_ohlcv_from_yfinance(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period)
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_ohlcv_from_db(self, security_id: int, 
                           start_date: datetime, 
                           end_date: datetime) -> pd.DataFrame:
        with self.db_manager.get_connection("price") as conn:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv
                WHERE security_id = ? AND timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            """
            df = pd.read_sql_query(query, conn, 
                                  params=(security_id, str(start_date), str(end_date)),
                                  parse_dates=['timestamp'],
                                  index_col='timestamp')
            return df
    
    def get_price_series(self, security_id: int, 
                        start_date: datetime, 
                        end_date: datetime,
                        price_type: str = 'close') -> pd.Series:
        df = self.fetch_ohlcv_from_db(security_id, start_date, end_date)
        if df.empty:
            return pd.Series()
        return df[price_type]

# ============================================================================
# QUANTZ SERVICE
# ============================================================================

class QuantzService:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.price_service = PriceService(db_manager)
        self.quantzlib = Quantzlib()
    
    def compute_beta_live(self, 
                         security_id: int,
                         reference_index_id: int,
                         lookback_period: LookbackPeriod,
                         rolling_window: RollingWindow) -> Dict:
        period_map = {
            LookbackPeriod.THREE_MONTH: 90,
            LookbackPeriod.TWO_YEAR: 730,
            LookbackPeriod.FIVE_YEAR: 1825
        }
        days = period_map[lookback_period]
        
        window_map = {
            RollingWindow.ONE_MONTH: 21,
            RollingWindow.TWO_MONTH: 42,
            RollingWindow.THREE_MONTH: 63
        }
        window_days = window_map[rolling_window]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 100)  # Extra buffer for rolling window
        
        stock_prices = self.price_service.get_price_series(
            security_id, start_date, end_date, 'close'
        )
        market_prices = self.price_service.get_price_series(
            reference_index_id, start_date, end_date, 'close'
        )
        
        print(f"DEBUG: Stock prices count: {len(stock_prices)}")
        print(f"DEBUG: Market prices count: {len(market_prices)}")
        
        if stock_prices.empty or market_prices.empty:
            raise ValueError(f"No price data available. Stock: {len(stock_prices)}, Market: {len(market_prices)}")
        
        stock_returns = self.quantzlib.calculate_returns(stock_prices)
        market_returns = self.quantzlib.calculate_returns(market_prices)
        
        print(f"DEBUG: Stock returns count: {len(stock_returns)}")
        print(f"DEBUG: Market returns count: {len(market_returns)}")
        
        # Align the data to same dates
        aligned_stock, aligned_market = stock_returns.align(market_returns, join='inner')
        
        print(f"DEBUG: Aligned returns count: {len(aligned_stock)}")
        
        if len(aligned_stock) < window_days:
            raise ValueError(f"Insufficient data for rolling window. Need {window_days}, have {len(aligned_stock)}")
        
        rolling_beta = self.quantzlib.calculate_beta(
            aligned_stock, aligned_market, window_days
        )
        
        overall_beta = self.quantzlib.calculate_beta(
            aligned_stock, aligned_market
        )
        
        print(f"DEBUG: Rolling beta count: {len(rolling_beta)}")
        print(f"DEBUG: Overall beta: {overall_beta}")
        
        return {
            "overall_beta": float(overall_beta),
            "rolling_beta": {str(k): float(v) for k, v in rolling_beta.items()},
            "stock_prices": {str(k): float(v) for k, v in stock_prices.items()},
            "market_prices": {str(k): float(v) for k, v in market_prices.items()},
            "timestamp": datetime.now().isoformat(),
            "data_points": len(aligned_stock)
        }
    
    def query_beta_from_db(self,
                          security_id: int,
                          start_date: datetime,
                          end_date: datetime,
                          pricebar_timeframe: Optional[PricebarTimeFrame] = None,
                          rolling_window: Optional[RollingWindow] = None) -> List[Dict]:
        with self.db_manager.get_connection("quantz") as conn:
            query = """
                SELECT * FROM beta
                WHERE security_id = ? AND timestamp BETWEEN ? AND ?
            """
            params = [security_id, str(start_date), str(end_date)]
            
            if pricebar_timeframe:
                query += " AND pricebar_timeframe = ?"
                params.append(pricebar_timeframe)
            
            if rolling_window:
                query += " AND rolling_window = ?"
                params.append(rolling_window)
            
            query += " ORDER BY timestamp"
            
            df = pd.read_sql_query(query, conn, params=params)
            return df.to_dict('records')

# ============================================================================
# QUANTZ SCRIPT (Cron Job Handler)
# ============================================================================

class QuantzScript:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.quantz_service = QuantzService(db_manager)
    
    def daily_beta_computation(self, 
                              security_id: int,
                              reference_index_id: int,
                              pricebar_timeframe: PricebarTimeFrame = PricebarTimeFrame.DAILY,
                              rolling_window: RollingWindow = RollingWindow.THREE_MONTH,
                              lookback_period: LookbackPeriod = LookbackPeriod.THREE_MONTH):
        try:
            result = self.quantz_service.compute_beta_live(
                security_id, reference_index_id, lookback_period, rolling_window
            )
            
            record = BetaRecord(
                timestamp=datetime.now(),
                security_id=security_id,
                pricebar_timeframe=pricebar_timeframe,
                rolling_window=rolling_window,
                reference_index="NIFTY50",
                lookback_period=lookback_period,
                value=result['overall_beta']
            )
            
            self.db_manager.store_beta(record)
            
            print(f"✓ Beta computed and stored for security {security_id}: {result['overall_beta']:.4f}")
            return True
            
        except Exception as e:
            print(f"✗ Error in daily beta computation: {str(e)}")
            return False

# ============================================================================
# FAST API APPLICATION
# ============================================================================

app = FastAPI(title="Quantz Analytics API", version="1.0.0", description="Quantitative Beta Analysis System")

db_manager = DatabaseManager()
quantz_service = QuantzService(db_manager)
quantz_script = QuantzScript(db_manager)

@app.get("/", response_class=HTMLResponse)
def root():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quantz Analytics Platform</title>
        <style>
            body { font-family: Arial; max-width: 1200px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; }
            .section { background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #667eea; }
            .method { display: inline-block; padding: 5px 10px; border-radius: 4px; font-weight: bold; color: white; margin-right: 10px; }
            .get { background: #28a745; }
            .post { background: #007bff; }
            button { background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background: #764ba2; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .success { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Quantz Analytics Platform</h1>
            <p>Quantitative Beta Computation & Analysis System</p>
        </div>
        
        <div class="section">
            <h2>Quick Start</h2>
            <button onclick="setupDemo()">Setup Demo Data</button>
            <button onclick="location.href='/docs'">API Documentation</button>
            <div id="status"></div>
        </div>
        
        <div class="section">
            <h2>📊 Available Endpoints</h2>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/v1/data/ingest</strong>
                <p>Ingest OHLCV data from Yahoo Finance</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/api/v1/beta/compute</strong>
                <p>Compute beta on-demand (Use Case 1: Live Analysis)</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/api/v1/beta/query</strong>
                <p>Query pre-computed beta from database (Use Case 2: Historical Data)</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span>
                <strong>/api/v1/beta/schedule</strong>
                <p>Schedule daily beta computation (Cron Job)</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/api/v1/visualization/beta</strong>
                <p>Get beta visualization data with rolling window</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/api/v1/visualization/chart</strong>
                <p>Interactive chart showing Beta vs Stock Price</p>
            </div>
        </div>
        
        <div class="section">
            <h2>📖 System Architecture</h2>
            <ul>
                <li><strong>Quantzlib:</strong> Core calculation library for statistical methods</li>
                <li><strong>Price Service:</strong> Manages OHLCV data from Yahoo Finance and Price DB</li>
                <li><strong>Quantz Service:</strong> Main service for beta computation and queries</li>
                <li><strong>Quantz Script:</strong> Handles scheduled cron jobs</li>
                <li><strong>Quantz DB:</strong> Stores computed beta values with metadata</li>
            </ul>
        </div>
        
        <script>
            async function setupDemo() {
                const status = document.getElementById('status');
                status.innerHTML = '<div class="status">Setting up demo data...</div>';
                
                try {
                    // Ingest Infosys data
                    await fetch('/api/v1/data/ingest?security_id=1&symbol=INFY.NS&period=1y', {method: 'POST'});
                    await fetch('/api/v1/data/ingest?security_id=999&symbol=^NSEI&period=1y', {method: 'POST'});
                    
                    status.innerHTML = '<div class="status success">✓ Demo data setup complete! Try the visualization endpoint.</div>';
                } catch(e) {
                    status.innerHTML = '<div class="status error">✗ Error: ' + e.message + '</div>';
                }
            }
        </script>
    </body>
    </html>
    """
    return html

@app.post("/api/v1/data/ingest")
def ingest_ohlcv_data(security_id: int, symbol: str, period: str = "1y"):
    try:
        price_service = PriceService(db_manager)
        df = price_service.fetch_ohlcv_from_yfinance(symbol, period)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        db_manager.store_ohlcv_data(security_id, df)
        
        return {
            "status": "success",
            "security_id": security_id,
            "symbol": symbol,
            "records": len(df),
            "date_range": f"{df.index[0]} to {df.index[-1]}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/beta/compute")
def compute_beta(
    security_id: int,
    reference_index_id: int = 999,
    lookback_period: LookbackPeriod = LookbackPeriod.THREE_MONTH,
    rolling_window: RollingWindow = RollingWindow.THREE_MONTH
):
    try:
        result = quantz_service.compute_beta_live(
            security_id, reference_index_id, lookback_period, rolling_window
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/beta/query")
def query_beta(
    security_id: int,
    start_date: str,
    end_date: str,
    pricebar_timeframe: Optional[PricebarTimeFrame] = None,
    rolling_window: Optional[RollingWindow] = None
):
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        results = quantz_service.query_beta_from_db(
            security_id, start, end, pricebar_timeframe, rolling_window
        )
        return {"data": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/beta/schedule")
def schedule_beta_computation(
    security_id: int,
    reference_index_id: int = 999
):
    try:
        success = quantz_script.daily_beta_computation(security_id, reference_index_id)
        return {
            "status": "success" if success else "failed",
            "message": "Beta computation completed",
            "security_id": security_id,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/visualization/beta")
def get_beta_visualization_data(
    security_id: int,
    months: int = 3,
    rolling_window: RollingWindow = RollingWindow.THREE_MONTH
):
    try:
        lookback = LookbackPeriod.THREE_MONTH if months == 3 else LookbackPeriod.TWO_YEAR
        
        result = quantz_service.compute_beta_live(
            security_id, 999, lookback, rolling_window
        )
        
        return {
            "beta_series": result['rolling_beta'],
            "stock_prices": result['stock_prices'],
            "market_prices": result['market_prices'],
            "overall_beta": result['overall_beta'],
            "rolling_window": rolling_window,
            "months": months
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/visualization/chart", response_class=HTMLResponse)
def get_visualization_chart(security_id: int = 1, months: int = 3):
    try:
        result = quantz_service.compute_beta_live(
            security_id, 999, LookbackPeriod.THREE_MONTH, RollingWindow.THREE_MONTH
        )
        
        dates = list(result['rolling_beta'].keys())
        betas = list(result['rolling_beta'].values())
        prices = [result['stock_prices'].get(d, 0) for d in dates]
        
        # Check if data exists
        if not dates or not betas:
            return f"""
            <html>
            <head><title>No Data Available</title></head>
            <body style="font-family: Arial; padding: 50px; text-align: center;">
                <h1>❌ No Data Available</h1>
                <p>Data points found: {len(dates)}, Beta values: {len(betas)}</p>
                <p>Please ingest data first before viewing visualization.</p>
                <h2>Steps to fix:</h2>
                <ol style="text-align: left; max-width: 600px; margin: 20px auto;">
                    <li>Go to: <a href="http://localhost:8000">http://localhost:8000</a></li>
                    <li>Click "Setup Demo Data" button</li>
                    <li>Wait 10-20 seconds for data download</li>
                    <li>Return to this page</li>
                </ol>
                <p>Or use command line:</p>
                <pre style="background: #f5f5f5; padding: 15px; text-align: left; max-width: 600px; margin: 20px auto;">
curl -X POST "http://localhost:8000/api/v1/data/ingest?security_id=1&symbol=INFY.NS&period=1y"
curl -X POST "http://localhost:8000/api/v1/data/ingest?security_id=999&symbol=^NSEI&period=1y"
                </pre>
                <button onclick="location.href='/'" style="padding: 10px 20px; font-size: 16px; margin-top: 20px;">Go to Homepage</button>
            </body>
            </html>
            """
        
        # Ensure we have valid min/max values
        min_beta = min(betas) if betas else 0
        max_beta = max(betas) if betas else 1
        
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Beta Visualization</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
                h1 {{ color: #667eea; }}
                .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
                .stat {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                .stat-value {{ font-size: 32px; font-weight: bold; }}
                .stat-label {{ font-size: 14px; opacity: 0.9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Beta vs Stock Price Analysis</h1>
                <p>Security ID: {security_id} | Period: {months} months | Rolling Window: 3 Month</p>
                
                <div class="stats">
                    <div class="stat">
                        <div class="stat-value">{result['overall_beta']:.3f}</div>
                        <div class="stat-label">Overall Beta</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value">{len(dates)}</div>
                        <div class="stat-label">Data Points</div>
                    </div>
                    <div class="stat">
                    <div class="stat-value">{min_beta:.3f}</div>
                        <div class="stat-label">Min Beta</div>
                    </div>
                    <div class="stat">
                    <div class="stat-value">{max_beta:.3f}</div>
                        <div class="stat-label">Max Beta</div>
                    </div>
                </div>
                
                <div id="chart"></div>
            </div>
            
            <script>
                var trace1 = {{
                    x: {dates},
                    y: {betas},
                    name: 'Rolling Beta',
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#667eea', width: 3 }},
                    yaxis: 'y'
                }};
                
                var trace2 = {{
                    x: {dates},
                    y: {prices},
                    name: 'Stock Price',
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#f093fb', width: 2 }},
                    yaxis: 'y2'
                }};
                
                var layout = {{
                    title: 'Rolling Beta (3M) vs Stock Price',
                    height: 600,
                    xaxis: {{ title: 'Date' }},
                    yaxis: {{ title: 'Beta', side: 'left' }},
                    yaxis2: {{
                        title: 'Stock Price',
                        overlaying: 'y',
                        side: 'right'
                    }},
                    hovermode: 'x unified',
                    plot_bgcolor: '#f8f9fa',
                    paper_bgcolor: 'white'
                }};
                
                Plotly.newPlot('chart', [trace1, trace2], layout);
            </script>
        </body>
        </html>
        """
        return html
    except Exception as e:
        return f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>"

# ============================================================================
# INITIALIZE & RUN
# ============================================================================

def initialize_system():
    print("\n" + "="*60)
    print("🚀 QUANTZ ANALYTICS PLATFORM")
    print("="*60)
    
    # Add sample securities
    db_manager.add_security(1, "INFY.NS", "Infosys")
    db_manager.add_security(2, "TCS.NS", "TCS")
    db_manager.add_security(3, "RELIANCE.NS", "Reliance Industries")
    db_manager.add_security(999, "^NSEI", "NIFTY 50")
    
    print("\n✓ Database initialized")
    print("✓ Sample securities added")
    print("\n" + "="*60)
    print("SERVER STARTING...")
    print("="*60)
    print("\n📍 Access the platform:")
    print("   → Main Interface: http://localhost:8000")
    print("   → API Docs: http://localhost:8000/docs")
    print("   → Visualization: http://localhost:8000/api/v1/visualization/chart?security_id=1")
    print("\n💡 Quick Start:")
    print("   1. Visit http://localhost:8000")
    print("   2. Click 'Setup Demo Data'")
    print("   3. Try the visualization endpoint")
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    initialize_system()
    uvicorn.run(app, host="0.0.0.0", port=8000)