from flask import Blueprint, flash

authBlueprint = Blueprint('auth', __name__, url_prefix="/auth")


@authBlueprint.route("/")
def authIndex():
    return "You have traversed to auth index"


@authBlueprint.route("/register", methods=["POST"])
def register():
    pass
