import os
import sys
import hashlib
import json
import logging
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import certifi
import time

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

log_timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_file = os.path.join(logs_dir, f"check_mongo_{log_timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CheckChanges")

HASH_FILE = "last_data_hash.txt"

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()

DATABASES = ["BACHV_BB_STOCKS", "BACHV_UD_STOCKS"]

def get_latest_collection(mongo_client, database):
    try:
        db = mongo_client[database]
        collections = [coll for coll in db.list_collection_names() if coll.startswith('Net_Data_')]
        if not collections:
            logger.warning(f"No collections found starting with 'Net_Data_' in database {database}")
            return None
        latest_collection = sorted(collections, reverse=True)[0]
        return latest_collection
    except Exception as e:
        logger.error(f"Error getting latest collection: {str(e)}")
        return None

def get_mongodb_data_hash():
    max_retries = 3
    retry_delay = 2
    
    for retry in range(max_retries):
        try:
            mongo_client = MongoClient(MONGO_DB_URL, tlsCAFile=ca, 
                                      serverSelectionTimeoutMS=30000,
                                      connectTimeoutMS=30000)
            mongo_client.admin.command('ping')
            all_data = []
            for database in DATABASES:
                latest_collection = get_latest_collection(mongo_client, database)
                if latest_collection is None:
                    continue
                db = mongo_client[database]
                collection = db[latest_collection]
                documents = list(collection.find({}, {'_id': 0}))
                all_data.append({
                    "database": database,
                    "collection": latest_collection,
                    "data": documents
                })
            data_string = json.dumps(all_data, sort_keys=True)
            data_hash = hashlib.md5(data_string.encode()).hexdigest()
            return data_hash
        except Exception as e:
            logger.error(f"Error getting data from MongoDB (attempt {retry+1}/{max_retries}): {str(e)}")
            if retry < max_retries - 1:
                logger.info(f"Trying to reconnect after {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Multiple connection attempts failed.")
                return None

def has_data_changed():
    current_hash = get_mongodb_data_hash()
    if current_hash is None:
        logger.error("Unable to get current data hash")
        return False
    if not os.path.exists(HASH_FILE):
        logger.info("Hash file doesn't exist, this might be the first run")
        with open(HASH_FILE, 'w') as f:
            f.write(current_hash)
        return True
    with open(HASH_FILE, 'r') as f:
        previous_hash = f.read().strip()
    has_changed = current_hash != previous_hash
    if has_changed:
        logger.info(f"Data changes detected. Old hash: {previous_hash}, New hash: {current_hash}")
        with open(HASH_FILE, 'w') as f:
            f.write(current_hash)
    else:
        logger.info("No data changes detected")
    return has_changed

if __name__ == "__main__":
    try:
        result = has_data_changed()
        if result:
            print("YES_CHANGED", flush=True)
            logger.info("Data changes detected - YES_CHANGED")
        else:
            print("NO_CHANGED", flush=True)
            logger.info("No data changes detected - NO_CHANGED")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print("ERROR", flush=True)
