from functools import wraps
import os
from flask import jsonify, request
import jwt

from utils.appLogger import log


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # jwt is passed in the request header
        if 'x-auth-token' in request.headers:
            token = request.headers['x-auth-token']
        # return 401 if token is not passed
        if not token:
            return jsonify({'message': 'Token Missing'}), 401

        try:
            # decoding the payload to fetch the stored details
            log(token)
            data = jwt.decode(token, os.getenv(
                'JWT_SECRET'), algorithms=["HS256"])

        except Exception as e:
            log(e)
            return jsonify({
                'message': 'Invalid Token!'
            }), 401
        # returns the current logged in users contex to the routes
        return f(data, *args, **kwargs)

    return decorated
