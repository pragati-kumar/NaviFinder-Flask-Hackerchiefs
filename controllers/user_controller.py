from bson.objectid import ObjectId
from flask import jsonify
from models.user_model import User
from utils.appLogger import log


def getUser(body):

    try:
        user = User.objects.raw({"_id": ObjectId(body["_id"])}).first()

    except StopIteration:
        return None

    return user


def addFCMToken(user, body):

    try:
        user = User.objects.raw({"_id": ObjectId(user["_id"])}).first()
    except StopIteration:
        return jsonify({"message": "User not found"}), 404

    user.fcmToken = body["fcmToken"]

    newUser = user.save()

    return jsonify({"status": "success", "user": newUser.toDict()}), 200
