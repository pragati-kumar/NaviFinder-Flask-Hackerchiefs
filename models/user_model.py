from pymodm import fields, MongoModel


class User(MongoModel):

    phone = fields.CharField()
    email = fields.EmailField()
    password = fields.CharField()
    fcmToken = fields.CharField()

    def __str__(self) -> str:
        return f"User(id: {self._id}, phone: {self.phone}, email: {self.email}, password: {self.password}, fcmToken: {self.fcmToken})"
