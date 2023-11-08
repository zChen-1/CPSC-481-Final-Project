import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import random
import re

# Load the training data
data = pd.read_csv('training_data.csv')

# Load the responses
response_data = pd.read_csv('responses.csv')

# Create a dictionary to hold responses for each intent
response_dict = {}
for intent in response_data['intent'].unique():
    response_dict[intent] = response_data[response_data['intent'] == intent]['response'].tolist()

inputs = data['utterance']
labels = data['intent']

# Split data into training and test sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2, random_state=42)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model with the training data
model.fit(train_inputs, train_labels)


# Function to handle math problems
def handle_math(problem_statement):
    try:
        # Extract numbers from the problem statement
        numbers = [int(num) for num in re.findall(r'\b\d+\b', problem_statement)]
        if any(word in problem_statement for word in ['add', 'plus', 'sum', '+']):
            result = numbers[0] + sum(numbers[1:])
            return f"The answer would be {result}."
        elif any(word in problem_statement for word in ['sub', 'minus', 'subtract', '-']):
            result = numbers[0] - sum(numbers[1:])
            return f"The answer would be {result}."
    except Exception as e:
        return "There was a problem calculating that."
    return "I'm not sure how to answer that."


# Chat
def chat():
    print("Hello! How can I help you? Type 'quit' to exit.")
    while True:
        user_input = input("You : ").lower()
        if user_input == 'quit':
            print("Goodbye! See you next time!")
            break

        # Predict the intent
        predicted_intent = model.predict([user_input])[0]

        if predicted_intent == 'math':
            print(handle_math(user_input))
        elif predicted_intent in response_dict:
            print(random.choice(response_dict[predicted_intent]))
        else:
            print("I'm not sure how to respond to that.")


if __name__ == "__main__":
    chat()
