from functools import wraps
import os
from flask import jsonify, request
import jwt


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
            data = jwt.decode(token, os.getenv('JWT_SECRET'))

        except:
            return jsonify({
                'message': 'Invalid Token!'
            }), 401
        # returns the current logged in users contex to the routes
        return f(data, *args, **kwargs)

    return decorated
