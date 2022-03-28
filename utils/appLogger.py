import os
import inspect


def log(o):
    if os.getenv("FLASK_ENV") == "development":

        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)

        print("-------------------------------")
        print("🐛", f"[{calframe[1][1].split('SIH-2022-Flask/')[1]}]", o)
        print("-------------------------------")
