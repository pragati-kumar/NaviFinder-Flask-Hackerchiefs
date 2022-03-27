import os
from pymodm.connection import connect

from utils.appLogger import log

connect(os.getenv("MONGO_URL"))

log("Connected")
