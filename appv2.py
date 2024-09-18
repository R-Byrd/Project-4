from flask import Flask, render_template, request, jsonify
import lillyv2  # Import your chatbot file

app = Flask(__name__)

# Route for the index page
@app.route('/')
def index():
    return render_template('indexv2.html')

# API endpoint to handle chatbot interaction
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    
    # Get the chatbot response based on user input
    response = lillyv2.suggest_questions_based_on_model(user_input)
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
