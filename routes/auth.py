import os
from flask import Blueprint, flash, jsonify, request
import jwt
import bcrypt

from models.user_model import User
from utils.appLogger import log

authBlueprint = Blueprint('auth', __name__, url_prefix="/auth")


@authBlueprint.route("/")
def authIndex():
    return "You have traversed to auth index"


@authBlueprint.route("/register", methods=["POST"])
def register():

    body = request.json

    user = User.objects.raw({'phone': body["phone"]})

    # log(user.count())

    if user.count():
        return jsonify({"message": "User already exists"}), 403

    if body["confirmPassword"] != body["password"]:
        return jsonify({"message": "Passwords dont match"}), 400

    hashed = bcrypt.hashpw(
        body["password"].encode("utf-8"), bcrypt.gensalt(10))

    newUser = User(phone=body["phone"], password=str(hashed)[2:-1])
    log(newUser)

    user = newUser.save()

    jwtKey = jwt.encode(
        {"_id": str(user._id), "phone": user.phone}, os.getenv("JWT_SECRET"))

    return jsonify({**(user.toDict()), "jwtKey": jwtKey}), 201


@authBlueprint.route("/login", methods=["POST"])
def login():

    body = request.json

    # log(body)

    try:
        user = User.objects.raw({'phone': body["phone"]}).first()

    except User.DoesNotExist:
        return jsonify({"message": "User does not exist"}), 404

    # log(type(user.password))

    isMatch = bcrypt.checkpw(
        body["password"].encode("utf-8"), user.password.encode("utf-8"))

    if not isMatch:
        return jsonify({"message": "Password does not match"}), 401

    jwtKey = jwt.encode(
        {"_id": str(user._id), "phone": user.phone}, os.getenv("JWT_SECRET"))

    return jsonify({**(user.toDict()), "jwtKey": jwtKey}), 200
