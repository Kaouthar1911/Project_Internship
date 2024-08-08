from flask import Flask
from flask_jwt_extended import JWTManager
from config import Config
from .services.model_manager import ModelManager
from .services.index_manager import IndexManager
from .services.chatbot_engine import ChatbotEngine
from .services.admin_manager import AdminManager

jwt = JWTManager()
model_manager = ModelManager()
index_manager = IndexManager(model_manager)
chatbot_engine = ChatbotEngine(index_manager)
admin_manager = AdminManager(index_manager)

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    jwt.init_app(app)

    from App.routes import auth, chatbot, admin
    app.register_blueprint(auth.bp)
    app.register_blueprint(chatbot.bp)
    app.register_blueprint(admin.bp)

    return app