from dotenv import load_dotenv
from __init__ import create_app
load_dotenv()

import config.database

app = create_app(__name__)


@app.route("/")
def index():
    return "Server running successfully"


if __name__ == "__main__":
    app.run(port=4000)
