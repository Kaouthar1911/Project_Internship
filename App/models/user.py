from werkzeug.security import generate_password_hash, check_password_hash

class User:
    def __init__(self, username, password, is_admin=False):
        self.username = username
        self.password_hash = generate_password_hash(password)
        self.is_admin = is_admin

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Simulating a database with a dictionary
users = {
    "admin": User("admin", "admin_password", True)
}