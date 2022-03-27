import os
from pymodm.connection import connect

connect(os.getenv("MONGO_URL"), alias="db")
