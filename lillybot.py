import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV containing questions, answers, and categories (adjust the file path as necessary)
faq_df = pd.read_csv('resources/answers.csv')  # Replace with the correct file path

# Combine both questions and answers into a single column for analysis
faq_df['Combined'] = faq_df['Questions'] + " " + faq_df['Answers']

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Vectorize the combined questions and answers
X = vectorizer.fit_transform(faq_df['Combined'])

# Use 'Category' as labels (Y)
Y = faq_df['Category']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate the model's accuracy
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

def suggest_questions_based_on_model(user_input):
    # Detect if the user is asking for the location of Antigua or Barbuda
    if "where" in user_input.lower() and ("antigua" in user_input.lower() or "barbuda" in user_input.lower()):
        return [{
            'question': user_input,
            'answer': 'map'  # Special signal to display the map
        }]

    # Transform the user's input into the TF-IDF vector space
    user_input_vector = vectorizer.transform([user_input])
    
    # Predict the category using the trained model
    predicted_category = model.predict(user_input_vector)[0]
    
    # Filter FAQ by the predicted category
    category_faqs = faq_df[faq_df['Category'] == predicted_category]
    
    # Compute the cosine similarity between the user's input and the predefined questions + answers
    category_faq_vectors = vectorizer.transform(category_faqs['Combined'])
    similarities = cosine_similarity(user_input_vector, category_faq_vectors)
    
    # Sort the questions based on the similarity scores in descending order
    similar_indices = similarities.argsort()[0][::-1]
    
    # Get the top 3 most similar questions
    top_3_indices = similar_indices[:3]
    
    suggested_questions = category_faqs['Questions'].iloc[top_3_indices]
    suggested_answers = category_faqs['Answers'].iloc[top_3_indices]
    
    # Format suggestions as a dictionary to return
    suggestions = [{'question': q, 'answer': a} for q, a in zip(suggested_questions, suggested_answers)]
    
    return suggestions
