from time import time
from flask import Blueprint, request, jsonify

from middlewares.auth_middleware import token_required
from mlScripts.outdoor import processCoordinates
from mlScripts.indoor import getCoordinates
from utils.appLogger import log

locationBlueprint = Blueprint("location", __name__, url_prefix="/location")


@locationBlueprint.route("/")
def locationIndex():
    return "Location Index"


@locationBlueprint.route("/outdoor", methods=["POST"])
@token_required
def getOutdoorLocation(user):

    body = request.json

    res = processCoordinates(
        user["_id"], "phone", time() * 1000, body["latitude"], body["longitude"], "phone", body["trial"])

    log(res)

    latitude, longitude = res

    return jsonify({"latitude": latitude, "longitude": longitude}), 200

@locationBlueprint.route("/indoor", methods=["POST"])
@token_required
def getIndoorLocation(user):

    body = request.json

    res = getCoordinates(
        user["_id"], body["rssi"], "mlScripts/learn_model_dict0.pth" , "mlScripts/ss_scaler.save", body["trial"])

    log(res)

    x, y, floor = res

    return jsonify({"x": None if x is None else float(x), "y": None if y is None else float(y), "floor": None if floor is None else  floor}), 200
