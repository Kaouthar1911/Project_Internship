from flask import Blueprint, request, jsonify, render_template, redirect, url_for
from flask_jwt_extended import create_access_token, set_access_cookies
from App.models.user import users

bp = Blueprint('auth', __name__)

@bp.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = users.get(username)
        if user and user.check_password(password) and user.is_admin:
            access_token = create_access_token(identity=username)
            response = redirect(url_for('admin.dashboard'))
            set_access_cookies(response, access_token)
            return response
        
        return render_template('admin_login.html', error="Invalid credentials"), 401
    
    return render_template('admin_login.html')