from dotenv import load_dotenv
from flask import Flask
load_dotenv()

app = Flask(__name__)

import config.database


@app.route("/")
def index():
    return "Server running successfully"
