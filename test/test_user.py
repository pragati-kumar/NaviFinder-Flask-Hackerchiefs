import unittest
import sys
sys.path.append('../')
from app import app
import json
import pymongo
import os



class UserTests(unittest.TestCase):
    def setUp(self):
        self.ctx = app.app_context()
        self.ctx.push()
        self.client = app.test_client()

    def tearDown(self):
        self.ctx.pop()

    def test_registration_mismatch(self):
        response = self.client.post(
            "/auth/register", json={"phone": "987658235", "password": '123', "confirmPassword": '456'})
        response_text = json.loads(response.text)
        assert response_text['message'] == 'Passwords dont match'
        assert response.status_code == 400

    def test_registration(self):
        response = self.client.post(
            "/auth/register", json={"phone": "9876543210", "password": '123', "confirmPassword": '123'})
        response_text = json.loads(response.text)
        assert 'jwtKey' in response_text
        assert response.status_code == 201

    def test_reregistration(self):
        response = self.client.post(
            "/auth/register", json={"phone": "9876543210", "password": '123', "confirmPassword": '123'})
        response_text = json.loads(response.text)
        assert response_text['message'] == 'User already exists'
        assert response.status_code == 403

    def test_tlogin_wrong_password(self):
        response = self.client.post(
            "/auth/login", json={"phone": "9876543210", "password": '456'})
        response_text = json.loads(response.text)
        assert response_text['message'] == 'Password does not match'
        assert response.status_code == 401

    def test_tlogin_no_user(self):
        response = self.client.post(
            "/auth/login", json={"phone": "8976543210", "password": '123'})
        response_text = json.loads(response.text)
        assert response_text['message'] == 'User does not exist'
        assert response.status_code == 404

    def test_tlogin(self):
        response = self.client.post(
            "/auth/login", json={"phone": "9876543210", "password": '123'})
        response_text = json.loads(response.text)
        assert 'jwtKey' in response_text
        assert response.status_code == 200


if __name__ == "__main__":

    myclient = pymongo.MongoClient(os.getenv("MONGO_URL"))
    mydb = myclient[os.getenv("COLLECTIONS_NAME")]
    mycol = mydb[os.getenv("DB_NAME")]

    myquery = {"phone": "9876543210"}

    mycol.delete_one(myquery)
    unittest.main()
