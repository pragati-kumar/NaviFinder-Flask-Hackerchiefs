from time import time
from flask import Blueprint, request, jsonify

from middlewares.auth_middleware import token_required
from mlScripts.outdoor import processCoordinates

locationBlueprint = Blueprint("location", __name__, url_prefix="/location")


@locationBlueprint.route("/")
def locationIndex():
    return "Location Index"


@locationBlueprint.route("/outdoor", methods=["POST"])
@token_required
def getOutdoorLocation(user):

    body = request.json

    latitude, longitude = processCoordinates(
        user["_id"], "phone", time() * 1000, body["latitude"], body["longitude"], "phone", body["trial"])

    return jsonify({"latitude": latitude, "longitude": longitude}), 200
