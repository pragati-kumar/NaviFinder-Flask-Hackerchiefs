import os
import unittest
import sys
sys.path.append('../')
from app import app


class BasicTests(unittest.TestCase):
    def setUp(self):
        self.ctx = app.app_context()
        self.ctx.push()
        self.client = app.test_client()

    def tearDown(self):
        self.ctx.pop()

    def test_home(self):
        response = self.client.get("/")

        assert response.status_code == 200
        assert response.text == "Server running successfully"


if __name__ == "__main__":
    unittest.main()
