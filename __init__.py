from dotenv import load_dotenv
from flask import Flask

load_dotenv()


def create_app(name):
    app = Flask(name)

    from config import database

    from routes.auth import authBlueprint
    from routes.user import userBlueprint

    app.register_blueprint(authBlueprint)
    app.register_blueprint(userBlueprint)

    return app
