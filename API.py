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

@app.get("/stock-all-models")
def get_stock_all_models(
    ticker: str = Query(..., description="Stock ticker symbol (Required)"),
    market_state: str = Query(..., description="Market State: BB or UD (Required)"),
    month_year: str = Query(None, description="Month-Year format (e.g., 2024-03). Optional - if not provided, returns all dates")
):
    try:
        if market_state.upper() not in ["BB", "UD"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid market_state: {market_state}. Must be either 'BB' or 'UD'"
            )
        
        if not ticker or ticker.isspace():
            raise HTTPException(
                status_code=400,
                detail="Ticker cannot be empty"
            )
        
        latest_file = get_latest_csv(market_state)
        logger.info(f"Reading file for ticker {ticker}, month_year {month_year}, market_state {market_state}")
        
        df = pd.read_csv(latest_file)
        logger.info(f"Data loaded. Columns: {df.columns.tolist()}")
        
        required_columns = ['Ticker', 'Model', 'Month-Year', 'Index', 'Actual', 
                          'Prediction', 'Prob_Class_0', 'Prob_Class_1', 'Correct']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")

        if ticker not in df['Ticker'].unique():
            raise HTTPException(
                status_code=404,
                detail=f"Ticker '{ticker}' not found in {market_state} data"
            )

        if month_year:
            filtered_df = df[
                (df['Ticker'] == ticker) & 
                (df['Month-Year'] == month_year)
            ]
        else:
            filtered_df = df[df['Ticker'] == ticker]
        
        if filtered_df.empty:
            error_msg = f"No data found for Ticker={ticker}"
            if month_year:
                error_msg += f", Month-Year={month_year}"
            error_msg += f", Market-State={market_state}"
            raise HTTPException(status_code=404, detail=error_msg)
        
        result_df = filtered_df[required_columns]
        
        models = sorted(filtered_df['Model'].unique().tolist())
        
        model_statistics = {}
        for model in models:
            model_df = filtered_df[filtered_df['Model'] == model]
            total_pred = len(model_df)
            correct_pred = model_df['Correct'].sum()
            accuracy = (correct_pred / total_pred * 100) if total_pred > 0 else 0
            
            model_statistics[model] = {
                "total_predictions": total_pred,
                "correct_predictions": int(correct_pred),
                "accuracy": float(accuracy),
                "dates": sorted(model_df['Month-Year'].unique().tolist()),
                "total_dates": len(model_df['Month-Year'].unique())
            }
        
        total_predictions = len(result_df)
        correct_predictions = result_df['Correct'].sum()
        accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
        
        return JSONResponse(
            content={
                "status": "success",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "market_state": market_state.upper(),
                "query_params": {
                    "ticker": ticker,
                    "month_year": month_year if month_year else "all",
                },
                "available_models": models,
                "total_models": len(models),
                "overall_statistics": {
                    "total_predictions": total_predictions,
                    "correct_predictions": int(correct_predictions),
                    "accuracy": float(accuracy)
                },
                "model_statistics": model_statistics,
                "data": result_df.to_dict('records')
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_stock_all_models for {market_state}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/latest-date-all-ticker-data")
def get_latest_date_all_ticker_data(
    market_state: str = Query(..., description="Market State: BB or UD (Required)")
):
    try:
        # Validate market_state
        if market_state.upper() not in ["BB", "UD"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid market_state: {market_state}. Must be either 'BB' or 'UD'"
            )
        
        latest_file = get_latest_csv(market_state)
        logger.info(f"Reading file for market_state {market_state}")
        
        df = pd.read_csv(latest_file)
        logger.info(f"Data loaded. Columns: {df.columns.tolist()}")
        
        required_columns = ['Ticker', 'Model', 'Month-Year', 'Index', 'Actual', 
                          'Prediction', 'Prob_Class_0', 'Prob_Class_1', 'Correct']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in CSV: {missing_columns}")
        
        df = df.replace({pd.NA: None, pd.NaT: None, float('nan'): None})
        
        sample_date = df['Month-Year'].iloc[0]
        logger.info(f"Sample date format: {sample_date}")
        
        df['Date_Obj'] = pd.to_datetime(df['Month-Year'], format='mixed', errors='coerce')

        df['Year_Month'] = df['Date_Obj'].dt.strftime('%Y-%m')
        df['Year_Month_Obj'] = pd.to_datetime(df['Year_Month'], format='%Y-%m')

        latest_data = []
        tickers = df['Ticker'].unique()
        
        for ticker in tickers:
            ticker_df = df[df['Ticker'] == ticker]
            latest_date = ticker_df['Year_Month_Obj'].max()
            
            if pd.isna(latest_date):
                logger.warning(f"No valid date found for ticker {ticker}")
                continue
                
            latest_ticker_data = ticker_df[ticker_df['Year_Month_Obj'] == latest_date]
            
            if latest_ticker_data.empty:
                logger.warning(f"No data found for ticker {ticker} at date {latest_date}")
                continue
                
            latest_data.append({
                'ticker': ticker,
                'latest_date': latest_date.strftime('%Y-%m'),
                'models': []
            })
            for model in latest_ticker_data['Model'].unique():
                model_data = latest_ticker_data[latest_ticker_data['Model'] == model]
                
                if model_data.empty:
                    continue
                    
                prediction = model_data['Prediction'].iloc[0]
                Prob_Class_1 = model_data['Prob_Class_1'].iloc[0]
                Prob_Class_0 = model_data['Prob_Class_0'].iloc[0]
                index = model_data['Index'].iloc[0]
                
                latest_data[-1]['models'].append({
                    'model_name': model,
                    'index': int(index) if index is not None else None,
                    'prediction': int(prediction) if prediction is not None else None,
                    'Prob_Class_1': float(Prob_Class_1) if Prob_Class_1 is not None else None,
                    'Prob_Class_0': float(Prob_Class_0) if Prob_Class_0 is not None else None,
               

                    })
        return JSONResponse(
            content={
                "status": "success",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "market_state": market_state.upper(),
                "total_tickers": len(latest_data),
                "data": latest_data
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_latest_ticker_data for {market_state}: {str(e)}")
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
