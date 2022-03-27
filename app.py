from __init__ import create_app

app = create_app(__name__)


@app.route("/")
def index():
    return "Server running successfully"


if __name__ == "__main__":
    app.run(port=4000)
