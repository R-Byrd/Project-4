from flask import Flask, render_template, request, jsonify
import lillybotv1  # Import your chatbot file

app = Flask(__name__)

# Route for the index page
@app.route('/')
def index():
    return render_template('indexv1.html')

# API endpoint to handle chatbot interaction
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    # Get suggested questions and answers based on user input
    suggestions = lillybotv1.suggest_questions_based_on_model(user_input)
    
    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    app.run(debug=True)