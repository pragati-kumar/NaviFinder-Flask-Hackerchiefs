import os


def log(o):
    if os.getenv("FLASK_ENV") == "development":
        print("-------------------------------")
        print("🐛", o)
        print("-------------------------------")
