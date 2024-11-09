from flask import Flask, request, jsonify, render_template
from bot import ChatAgentWithRL  # Import the chatbot class

# Initialize the Flask app and ChatAgentWithRL instance
app = Flask(__name__)
chat_agent = ChatAgentWithRL()

# Define the home route to render the main chat page
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML template

# Define a route for chatting with the bot
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("query")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Get response from the chat agent
    response = chat_agent.chat(user_input)

    return jsonify({"response": response})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    user_input = data.get('user_input')
    response = data.get('response')
    feedback = data.get('feedback')

    # Assuming `feedback` isn't used in `train_rl`, remove it from the call
    chat_agent.train_rl(user_input, response)

    return jsonify({"status": "Feedback received"})

if __name__ == '__main__':
    app.run(debug=True)
