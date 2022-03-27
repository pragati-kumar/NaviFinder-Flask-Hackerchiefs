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

    if body["confirmPassword"] != body["password"]:
        return jsonify({"message": "Passwords dont match"}), 400

    hashed = bcrypt.hashpw(
        body["password"].encode("utf-8"), bcrypt.gensalt(10))

    newUser = User(phone=body["phone"], password=hashed)
    log(newUser)

    user = newUser.save()

    jwtKey = jwt.encode(
        {"_id": str(user._id), "phone": user.phone}, os.getenv("JWT_SECRET"))

    return jsonify({**(user.toDict()), "jwtKey": jwtKey}), 201
