from flask import Blueprint, request, jsonify
from controllers import user_controller

from middlewares.auth_middleware import token_required

userBlueprint = Blueprint("user", __name__, url_prefix="/users")


@userBlueprint.route("/")
def userIndex():
    return "You have come to User Index Route"


@userBlueprint.route("/my", methods=["GET", "PUT"])
@token_required
def updateUser(user):

    if(request.method == "GET"):

        user = user_controller.getUser(user)

        if user is None:
            return jsonify({"message": "User does not exist"}), 404

        return jsonify({**user.toDict()}), 200

    if(request.method == "PUT"):
        pass
