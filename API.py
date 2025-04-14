import os
import glob
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
from datetime import datetime
from pydantic import BaseModel
import logging
import traceback

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

log_timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_file = os.path.join(logs_dir, f"api_{log_timestamp}.log")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified Stock Data API",
    description="API to get latest stock predictions data for both BB and UD models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

class StockData(BaseModel):
    date: str
    symbol: str
    prediction: float

class StockQuery(BaseModel):
    ticker: str
    model: str
    month_year: str
    data_type: str

def get_latest_csv(data_type="BB"):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        if data_type.upper() not in ["BB", "UD"]:
            raise ValueError(f"Invalid data_type: {data_type}. Must be 'BB' or 'UD'")
            
        sub_dir = data_type.upper()
        data_dir = os.path.join(current_dir, "Get_Data", sub_dir)
        
        if not os.path.exists(data_dir):
            logger.error(f"Directory not found: {data_dir}")
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        pattern = os.path.join(data_dir, "mongodb_data_*.csv")
        csv_files = glob.glob(pattern)
        
        logger.info(f"Found CSV files in {sub_dir}: {csv_files}")
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {sub_dir} directory")
        
        latest_file = max(csv_files, key=os.path.getctime)
        logger.info(f"Latest CSV file for {sub_dir}: {latest_file}")
        
        if not os.path.isfile(latest_file):
            raise FileNotFoundError(f"File not found: {latest_file}")
            
        df = pd.read_csv(latest_file)
        logger.info(f"File loaded successfully. Columns: {df.columns.tolist()}")
        
        return latest_file
    except Exception as e:
        logger.error(f"Error in get_latest_csv for {data_type}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

@app.get("/test")
async def test_endpoint(data_type: str = Query("BB", description="Market State: BB or UD")):
    try:
        latest_file = get_latest_csv(data_type)
        df = pd.read_csv(latest_file)
        return {
            "status": "success",
            "data_type": data_type,
            "file": latest_file,
            "rows": len(df),
            "columns": list(df.columns)
        }
    except Exception as e:
        logger.error(f"Error in test endpoint for {data_type}: {str(e)}")
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e)}

@app.get("/stock-prediction")
def get_stock_prediction(
    ticker: str = Query(..., description="Stock ticker symbol"),
    model: str = Query(..., description="Model name"),
    month_year: str = Query(..., description="Month-Year format (e.g., 2024-03)"),
    data_type: str = Query("BB", description="Market State: BB or UD")
):
    try:
        latest_file = get_latest_csv(data_type)
        logger.info(f"Reading file for ticker {ticker}, model {model}, month-year {month_year}, data_type {data_type}")
        
        df = pd.read_csv(latest_file)
        logger.info(f"Data loaded. Columns: {df.columns.tolist()}")
        
        required_columns = ['Ticker', 'Model', 'Month-Year', 'Index', 'Actual', 
                          'Prediction', 'Prob_Class_0', 'Prob_Class_1', 'Correct']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")
        
        filtered_df = df[
            (df['Ticker'] == ticker) & 
            (df['Model'] == model) & 
            (df['Month-Year'] == month_year)
        ]
        
        if filtered_df.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for Ticker={ticker}, Model={model}, Month-Year={month_year}, Data-Type={data_type}"
            )
        
        result_df = filtered_df[required_columns]
        
        total_predictions = len(result_df)
        correct_predictions = result_df['Correct'].sum()
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        return JSONResponse(
            content={
                "status": "success",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_type": data_type,
                "query_params": {
                    "ticker": ticker,
                    "model": model,
                    "month_year": month_year
                },
                "statistics": {
                    "total_predictions": total_predictions,
                    "correct_predictions": int(correct_predictions),
                    "accuracy": float(accuracy)
                },
                "data": result_df.to_dict('records')
            }
        )
    except Exception as e:
        logger.error(f"Error in get_stock_prediction for {data_type}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-filters")
def get_available_filters(data_type: str = Query("BB", description="Market State: BB or UD")):
    try:
        latest_file = get_latest_csv(data_type)
        df = pd.read_csv(latest_file)
        
        return {
            "data_type": data_type,
            "tickers": sorted(df['Ticker'].unique().tolist()),
            "models": sorted(df['Model'].unique().tolist()),
            "month_years": sorted(df['Month-Year'].unique().tolist())
        }
    except Exception as e:
        logger.error(f"Error in get_available_filters for {data_type}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stock-all-predictions")
def get_stock_all_predictions(
    ticker: str = Query(..., description="Stock ticker symbol"),
    model: str = Query(..., description="Model name"),
    data_type: str = Query("BB", description="Market State: BB or UD")
):
    try:
        latest_file = get_latest_csv(data_type)
        logger.info(f"Reading file for ticker {ticker}, model {model}, data_type {data_type}")
        
        df = pd.read_csv(latest_file)
        logger.info(f"Data loaded. Columns: {df.columns.tolist()}")
        
        required_columns = ['Ticker', 'Model', 'Month-Year', 'Index', 'Actual', 
                          'Prediction', 'Prob_Class_0', 'Prob_Class_1', 'Correct']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")
        
        # Lọc dữ liệu theo ticker và model
        filtered_df = df[
            (df['Ticker'] == ticker) & 
            (df['Model'] == model)
        ]
        
        if filtered_df.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data found for Ticker={ticker}, Model={model}, Data-Type={data_type}"
            )
        
        result_df = filtered_df[required_columns]
        
        # Tính toán thống kê tổng thể
        total_predictions = len(result_df)
        correct_predictions = result_df['Correct'].sum()
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        return JSONResponse(
            content={
                "status": "success",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_type": data_type,
                "query_params": {
                    "ticker": ticker,
                    "model": model
                },
                "statistics": {
                    "total_predictions": total_predictions,
                    "correct_predictions": int(correct_predictions),
                    "accuracy": float(accuracy)
                },
                "dates": sorted(filtered_df['Month-Year'].unique().tolist()),
                "total_dates": len(filtered_df['Month-Year'].unique()),
                "data": result_df.to_dict('records')
            }
        )
    except Exception as e:
        logger.error(f"Error in get_stock_all_predictions for {data_type}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

def print_csv_files():
    logger.info("="*50)
    logger.info("CHECKING CSV FILES IN USE:")
    logger.info("="*50)
    
    try:
        bb_file = get_latest_csv("BB")
        logger.info(f"✅ BB FILE: {os.path.basename(bb_file)}")
        logger.info(f"   Full path: {bb_file}")
        
        bb_modified_time = datetime.fromtimestamp(os.path.getmtime(bb_file))
        logger.info(f"   Last updated: {bb_modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        bb_size = os.path.getsize(bb_file) / 1024
        logger.info(f"   Size: {bb_size:.2f} KB")
    except Exception as e:
        logger.error(f"❌ Error reading BB file: {str(e)}")
    
    logger.info("-"*50)
    
    try:
        ud_file = get_latest_csv("UD")
        logger.info(f"✅ UD FILE: {os.path.basename(ud_file)}")
        logger.info(f"   Full path: {ud_file}")
        
        ud_modified_time = datetime.fromtimestamp(os.path.getmtime(ud_file))
        logger.info(f"   Last updated: {ud_modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        ud_size = os.path.getsize(ud_file) / 1024
        logger.info(f"   Size: {ud_size:.2f} KB")
    except Exception as e:
        logger.error(f"❌ Error reading UD file: {str(e)}")
    
    logger.info("="*50)

if __name__ == "__main__":
    import uvicorn
    
    print_csv_files()
    
    logger.info("Starting Unified API server...")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
