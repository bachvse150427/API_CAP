import os
import sys
import pandas as pd
from datetime import datetime
import pymongo
from dotenv import load_dotenv
import certifi
import time

from ApiStock.exception.exception import ApiPredictionException
from ApiStock.logging.logger import logging

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

log_timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_file = os.path.join(logs_dir, f"mongo_data_{log_timestamp}.log")

file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logger = logging.getLogger('GetMongoData')
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()

class NetworkDataFetch:
    def __init__(self):
        max_retries = 3
        retry_delay = 2
        
        for retry in range(max_retries):
            try:
                self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca,
                                                      serverSelectionTimeoutMS=30000,
                                                      connectTimeoutMS=30000)
                self.mongo_client.admin.command('ping')
                break
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"MongoDB connection error (attempt {retry+1}/{max_retries}): {str(e)}")
                    logger.info(f"Trying to reconnect after {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise ApiPredictionException(e, sys)

    def get_latest_collection(self, database):
        try:
            db = self.mongo_client[database]
            collections = [coll for coll in db.list_collection_names() if coll.startswith('Net_Data_')]
            if not collections:
                logger.warning(f"No collections found starting with 'Net_Data_' in database {database}")
                logger.warning(f"Available collections: {db.list_collection_names()}")
                return None
            latest_collection = sorted(collections, reverse=True)[0]
            return latest_collection
        except Exception as e:
            raise ApiPredictionException(e, sys)

    def fetch_data_from_mongodb(self, database, collection):
        try:
            db = self.mongo_client[database]
            collection_data = db[collection]
            data = list(collection_data.find({}, {'_id': 0}))
            return data
        except Exception as e:
            raise ApiPredictionException(e, sys)

    def save_to_csv(self, data, output_path):
        try:
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            return output_path
        except Exception as e:
            raise ApiPredictionException(e, sys)

    def fetch_and_save_data(self, database, base_output_dir="Get_Data", filename_prefix="mongodb_data"):
        try:
            sub_dir = ""
            if "BB" in database:
                sub_dir = "BB"
            elif "UD" in database:
                sub_dir = "UD"
            else:
                sub_dir = "Other"
                
            output_dir = os.path.join(base_output_dir, sub_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(output_dir, f"{filename_prefix}_{current_time}.csv")
            
            latest_collection = self.get_latest_collection(database)
            
            if latest_collection is None:
                logger.error(f"Error: No collections found with the required prefix in {database}.")
                return None
                
            logger.info(f"Latest collection in {database}: {latest_collection}")
            
            data = self.fetch_data_from_mongodb(database, latest_collection)
            logger.info(f"Found {len(data)} records from {database}")
            
            if len(data) == 0:
                logger.warning(f"Warning: No data found in the collection from {database}.")
                return None
            
            saved_file = self.save_to_csv(data, output_file)
            logger.info(f"Data from {database} saved to: {saved_file}")
            return saved_file
        except Exception as e:
            raise ApiPredictionException(e, sys)

if __name__ == '__main__':
    DATABASES = ["BACHV_BB_STOCKS", "BACHV_UD_STOCKS"]
    OUTPUT_DIR = "Get_Data"
    
    logger.info("Starting MongoDB data fetch process")
    network_fetch = NetworkDataFetch()
    
    for db in DATABASES:
        try:
            logger.info(f"Processing database: {db}")
            result = network_fetch.fetch_and_save_data(db, OUTPUT_DIR)
            if result:
                logger.info(f"Successfully fetched and saved data from {db}")
            else:
                logger.error(f"Failed to fetch data from {db}")
        except Exception as e:
            logger.error(f"Error processing {db}: {str(e)}")
    
    logger.info("MongoDB data fetch process completed")
