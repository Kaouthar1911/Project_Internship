from flask import Blueprint, request, jsonify, render_template
from App.services.chatbot_engine import ChatbotEngine

bp = Blueprint('chatbot', __name__)
chatbot_engine = ChatbotEngine()

@bp.route('/', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form.get('message', '')
        response = chatbot_engine.chat(user_input)
        return jsonify({"response": response}), 200
    return render_template('chatbot.html')