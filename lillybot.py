import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV containing questions and answers (adjust the file path as necessary)
faq_df = pd.read_csv('resources/answers.csv')  # Replace with the correct file path

# Combine both questions and answers into a single column for analysis
faq_df['Combined'] = faq_df['Questions'] + " " + faq_df['Answers']

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Vectorize the combined questions and answers
X = vectorizer.fit_transform(faq_df['Combined'])

def suggest_questions_based_on_model(user_input):
    # Detect if the user is asking for the location of Antigua or Barbuda
    if "where" in user_input.lower() and ("antigua" in user_input.lower() or "barbuda" in user_input.lower()):
        return [{
            'question': user_input,
            'answer': 'map'  # Special signal to display the map
        }]
    
    # Detect if the user is asking to view the interactive map
    if "interactive map" in user_input.lower():
        return [{
            'question': user_input,
            'answer': 'interactive_map'  # Special signal to display the interactive map
        }]
    
    # If not a special query, proceed with similarity matching
    user_input_vector = vectorizer.transform([user_input])
    
    # Compute the cosine similarity between the user's input and the predefined questions + answers
    similarities = cosine_similarity(user_input_vector, X)
    
    # Sort the questions based on the similarity scores in descending order
    similar_indices = similarities.argsort()[0][::-1]
    
    # Get the top 3 most similar questions
    top_3_indices = similar_indices[:3]
    
    suggested_questions = faq_df['Questions'].iloc[top_3_indices]
    suggested_answers = faq_df['Answers'].iloc[top_3_indices]
    
    # Format suggestions as a dictionary to return
    suggestions = [{'question': q, 'answer': a} for q, a in zip(suggested_questions, suggested_answers)]
    
    return suggestions

# Accuracy Evaluation Function
def evaluate_model_accuracy():
    correct_predictions = 0
    total = len(faq_df)
    
    for index, row in faq_df.iterrows():
        question = row['Questions']
        actual_answer = row['Answers']
        
        # Get the top suggestion for the current question
        suggested_answers = suggest_questions_based_on_model(question)
        top_answer = suggested_answers[0]['answer']  # The most similar answer
        
        # Check if the top predicted answer matches the actual answer
        if top_answer.strip() == actual_answer.strip():
            correct_predictions += 1
    
    # Calculate accuracy
    accuracy = correct_predictions / total * 100
    print(f"Model Accuracy: {accuracy:.2f}%")

# Test example
user_input = "Where is Antigua?"
suggestions = suggest_questions_based_on_model(user_input)
print(suggestions)

# Evaluate model accuracy
evaluate_model_accuracy()
