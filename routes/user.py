from flask import Blueprint, request, jsonify

userBlueprint = Blueprint("user", __name__, url_prefix="/users")


@userBlueprint.route("/")
def userIndex():
    return "You have come to User Index Route"
