from transformers import pipeline

# Initialize the question-answering pipeline
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

# Load the text data from a .txt file (answers.txt) in the resources folder
with open('resources/answers.txt', 'r', encoding='utf-8') as file:
    context = file.read()  # Read the entire content of the text file

# Function to handle special case queries like map requests
def suggest_questions_based_on_model(user_input):
    # If "interactive" and "map" are present, return the interactive map regardless of Antigua/Barbuda
    if "interactive" in user_input.lower() and "map" in user_input.lower():
        return "interactive_map"  # Return the interactive map signal

    # If "interactive" is not present but "map" with "antigua" or "barbuda" is, return the normal map
    if "map" in user_input.lower() and ("antigua" in user_input.lower() or "barbuda" in user_input.lower()):
        return "map"  # Return the normal map signal

    # Use the question-answering model for other queries
    result = question_answerer(question=user_input, context=context)

    return result['answer']  # Return the result from the model

