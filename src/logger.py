import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format="[%(asctime)s] %(levelname)s %(name)s:%(lineno)d - %(message)s",
        level=logging.INFO,
    )

logging.getLogger("werkzeug").setLevel(logging.WARNING)
