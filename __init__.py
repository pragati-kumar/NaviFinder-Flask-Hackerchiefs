from dotenv import load_dotenv
from flask import Flask

load_dotenv()


def create_app(name):
    app = Flask(name)

    from config import database

    from routes.auth import authBlueprint
    app.register_blueprint(authBlueprint)

    return app
