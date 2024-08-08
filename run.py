from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from werkzeug.security import check_password_hash, generate_password_hash
from App.services.model_manager import ModelManager
from App.services.index_manager import IndexManager
from App.services.chatbot_engine import ChatbotEngine
from App.services.admin_manager import AdminManager

from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from werkzeug.security import check_password_hash, generate_password_hash
from App.services.model_manager import ModelManager
from App.services.index_manager import IndexManager
from App.services.chatbot_engine import ChatbotEngine
from App.services.admin_manager import AdminManager
import os

# Changez cette ligne pour utiliser le chemin correct
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'App', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'App', 'static'))
App = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

App.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
App.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')

jwt = JWTManager(App)

# Simulating a user database
users = {
    "admin": {
        "password": generate_password_hash("admin_password"),
        "is_admin": True
    }
}

model_manager = ModelManager()
index_manager = IndexManager(model_manager)
chatbot_engine = ChatbotEngine(index_manager)
admin_manager = AdminManager(index_manager)

@App.route('/')
def home():
    return render_template('chatbot.html')

@App.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    response = chatbot_engine.chat(user_input)
    return jsonify({"response": response})

@App.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    users = {
    "admin": {
        "password": generate_password_hash(" "),
        "is_admin": True
    },
    "user1": {
        "password": generate_password_hash("user1_password"),
        "is_admin": False
    }
}
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = users.get(username)
        if user and check_password_hash(user['password'], password) and user['is_admin']:
            access_token = create_access_token(identity=username)
            return redirect(url_for('admin_dashboard'))
        return render_template('admin_login.html', error="Invalid credentials")
    return render_template('admin_login.html')

@App.route('/admin/dashboard')
@jwt_required()
def admin_dashboard():
    current_user = get_jwt_identity()
    if users.get(current_user, {}).get('is_admin'):
        documents = admin_manager.list_documents()
        return render_template('admin_dashboard.html', documents=documents)
    return jsonify({"msg": "Admin privileges required"}), 403

@App.route('/admin/add_document', methods=['POST'])
@jwt_required()
def add_document():
    current_user = get_jwt_identity()
    if users.get(current_user, {}).get('is_admin'):
        file_path = request.form.get('file_path')
        result = admin_manager.add_new_document(file_path)
        return jsonify({"result": result})
    return jsonify({"msg": "Admin privileges required"}), 403

@App.route('/admin/update_document', methods=['POST'])
@jwt_required()
def update_document():
    current_user = get_jwt_identity()
    if users.get(current_user, {}).get('is_admin'):
        filename = request.form.get('filename')
        new_content = request.form.get('new_content')
        result = admin_manager.update_document(filename, new_content)
        return jsonify({"result": result})
    return jsonify({"msg": "Admin privileges required"}), 403

@App.route('/admin/delete_document', methods=['POST'])
@jwt_required()
def delete_document():
    current_user = get_jwt_identity()
    if users.get(current_user, {}).get('is_admin'):
        filename = request.form.get('filename')
        result = admin_manager.delete_document(filename)
        return jsonify({"result": result})
    return jsonify({"msg": "Admin privileges required"}), 403

if __name__ == '__main__':
    App.run(debug=True)