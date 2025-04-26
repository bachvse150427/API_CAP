import os
import sys
import subprocess
import time
import logging
from datetime import datetime
import os.path
import socket
import psutil

logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

log_timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
log_file = os.path.join(logs_dir, f"{log_timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")

def is_api_server_running():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('0.0.0.0', 8080))
        sock.close()
        return result == 0
    except Exception as e:
        logger.error(f"Error checking API server: {str(e)}")
        return False

def get_api_server_pid():
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['name'] == 'python.exe' or proc.info['name'] == 'python':
                cmdline = proc.info['cmdline']
                if cmdline and len(cmdline) > 1 and 'API.py' in cmdline[1]:
                    return proc.info['pid']
        return None
    except Exception as e:
        logger.error(f"Error finding API server process ID: {str(e)}")
        return None

def check_mongodb_changes():
    logger.info("Checking for MongoDB data changes...")
    try:
        check_log_timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        startupinfo = None
        if sys.platform == 'win32':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        max_retries = 3
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        for retry in range(max_retries):
            try:
                process = subprocess.Popen(
                    [sys.executable, "check_mongo_changes.py"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    startupinfo=startupinfo,
                    env=env,
                    universal_newlines=True
                )
                stdout, stderr = process.communicate(timeout=60)
                if stdout:
                    logger.info(f"Output from check_mongo_changes.py: {stdout.strip()}")
                if stderr:
                    logger.warning(f"Stderr from check_mongo_changes.py: {stderr.strip()}")
                if "YES_CHANGED" in stdout:
                    logger.info("Changes detected in MongoDB data!")
                    return True
                elif "NO_CHANGED" in stdout:
                    logger.info("No changes detected in MongoDB data.")
                    return False
                elif "ERROR" in stdout:
                    logger.error("check_mongo_changes.py script reported an error")
                    if retry < max_retries - 1:
                        logger.info(f"Retrying (attempt {retry + 1})...")
                        time.sleep(2)
                        continue
                    return False
                else:
                    logger.error(f"Undefined result from check_mongo_changes.py: '{stdout}'")
                    if retry < max_retries - 1:
                        logger.info(f"Retrying (attempt {retry + 1})...")
                        time.sleep(2)
                        continue
                    return False
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout running check_mongo_changes.py (attempt {retry + 1})")
                process.kill()
                if retry < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
    except Exception as e:
        logger.error(f"Error checking MongoDB changes: {str(e)}")
        return False

def run_get_mongo_data():
    logger.info("====================================")
    logger.info("Starting MongoDB data fetch...")
    logger.info("====================================")
    try:
        startupinfo = None
        if sys.platform == 'win32':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        max_retries = 3
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        for retry in range(max_retries):
            try:
                process = subprocess.Popen(
                    [sys.executable, "get_mongo_data.py"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    startupinfo=startupinfo,
                    env=env,
                    universal_newlines=True
                )
                stdout, stderr = process.communicate(timeout=120)
                if stdout:
                    logger.info(f"Output from get_mongo_data.py:\n{stdout}")
                if stderr and any(error_keyword in stderr for error_keyword in ["Error:", "Exception:", "Traceback"]):
                    logger.error(f"Errors from get_mongo_data.py:\n{stderr}")
                elif stderr and stderr.strip():
                    logger.info(f"Additional output from get_mongo_data.py:\n{stderr}")
                
                if process.returncode == 0:
                    logger.info("Successfully fetched data from MongoDB!")
                    return True
                else:
                    logger.error(f"Failed to fetch data from MongoDB! Exit code: {process.returncode}")
                    if retry < max_retries - 1:
                        logger.info(f"Retrying (attempt {retry + 1})...")
                        time.sleep(2)
                        continue
                    return False
            except subprocess.TimeoutExpired:
                logger.error(f"Timeout running get_mongo_data.py (attempt {retry + 1})")
                process.kill()
                if retry < max_retries - 1:
                    time.sleep(2)
                    continue
                return False
    except Exception as e:
        logger.error(f"Error running get_mongo_data.py: {str(e)}")
        return False

def main():
    logger.info("Starting program...")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    api_running = is_api_server_running()
    api_pid = get_api_server_pid()
    api_process = None
    if api_running:
        logger.info(f"API server is already running (PID: {api_pid})")
    else:
        initial_data_success = run_get_mongo_data()
        if initial_data_success:
            startupinfo = None
            if sys.platform == 'win32':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            api_process = subprocess.Popen([sys.executable, "API.py"], startupinfo=startupinfo)
            logger.info(f"API server started! (PID: {api_process.pid})")
        else:
            logger.error("Could not fetch data from MongoDB. API server not started.")
            sys.exit(1)
    try:
        while True:
            sleep_time = int(os.environ.get('CHECK_INTERVAL', 7200))
            logger.info(f"Waiting {sleep_time} seconds before checking for changes...")
            time.sleep(sleep_time)
            if api_process is not None and api_process.poll() is not None:
                logger.warning("API server has stopped, restarting...")
                startupinfo = None
                if sys.platform == 'win32':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE
                api_process = subprocess.Popen([sys.executable, "API.py"], startupinfo=startupinfo)
                logger.info(f"API server restarted! (PID: {api_process.pid})")
            if check_mongodb_changes():
                logger.info("Changes detected, updating data...")
                data_success = run_get_mongo_data()
                if data_success:
                    logger.info("Data updated successfully!")
                else:
                    logger.error("Could not update data from MongoDB.")
            else:
                logger.info("No changes detected, keeping current data.")
    except KeyboardInterrupt:
        logger.info("Received exit signal, stopping program...")
        if api_process is not None:
            api_process.terminate()
            logger.info("API server stopped.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if api_process is not None:
            api_process.terminate()
    logger.info("Program ended.")

if __name__ == "__main__":
    main()
