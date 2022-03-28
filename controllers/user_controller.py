from bson.objectid import ObjectId
from models.user_model import User
from utils.appLogger import log


def getUser(body):

    try:
        user = User.objects.raw({"_id": ObjectId(body["_id"])}).first()

    except StopIteration:
        return None

    return user
