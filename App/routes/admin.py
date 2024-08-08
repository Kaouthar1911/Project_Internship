from flask import Blueprint, request, jsonify, render_template
from flask_jwt_extended import jwt_required, get_jwt_identity
from App.models.user import users
from App.services.admin_manager import AdminManager

bp = Blueprint('admin', __name__)
admin_manager = AdminManager()

def admin_required(fn):
    @jwt_required()
    def wrapper(*args, **kwargs):
        current_user = get_jwt_identity()
        user = users.get(current_user)
        if not user or not user.is_admin:
            return jsonify({"msg": "Admin privileges required"}), 403
        return fn(*args, **kwargs)
    return wrapper

@bp.route('/admin/dashboard', methods=['GET'])
@admin_required
def dashboard():
    documents = admin_manager.list_documents()
    return render_template('admin_dashboard.html', documents=documents)

@bp.route('/admin/add_document', methods=['POST'])
@admin_required
def add_document():
    file_path = request.form.get('file_path', '')
    result = admin_manager.add_new_document(file_path)
    return jsonify({"result": result}), 200

@bp.route('/admin/update_document', methods=['POST'])
@admin_required
def update_document():
    filename = request.form.get('filename', '')
    new_content = request.form.get('new_content', '')
    result = admin_manager.update_document(filename, new_content)
    return jsonify({"result": result}), 200

@bp.route('/admin/delete_document', methods=['POST'])
@admin_required
def delete_document():
    filename = request.form.get('filename', '')
    result = admin_manager.delete_document(filename)
    return jsonify({"result": result}), 200