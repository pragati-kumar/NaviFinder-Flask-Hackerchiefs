from pymodm import fields, MongoModel


class User(MongoModel):

    phone = fields.CharField()
    email = fields.EmailField()
    password = fields.CharField()
    fcmToken = fields.CharField()
