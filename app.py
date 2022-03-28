from __init__ import create_app

app = create_app(__name__)


@app.route("/")
def index():
    return "Server running successfully"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4000)
