from flask import Flask


def create_app(name):
    app = Flask(name)
    # existing code omitted

    from routes.auth import authBlueprint
    app.register_blueprint(authBlueprint)

    return app
