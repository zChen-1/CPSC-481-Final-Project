from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import random
import re

app = Flask(__name__)

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
# Limitation can only perform one type of operation
# e.g., 10 + 5 - 5 will return 20 (incorrect answer)
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
        elif any(word in problem_statement for word in ['divide','div', '/']):
            result = numbers[0]
            for i in range(1, len(numbers)):
                result = result / numbers[i]
            return f"The answer would be {result}"
        elif any(word in problem_statement for word in ['multiply', 'times', '*']):
            result = numbers[0]
            for i in range(1, len(numbers)):
                result = result * numbers[i]
            return f"The answer would be {result}."
    except Exception as e:
        return "There was a problem calculating that."
    return "I'm not sure how to answer that."


# Chat, return response
def chat(user_input):
    # Predict the intent
    predicted_intent = model.predict([user_input])[0]

    if predicted_intent == 'math':
        print(handle_math(user_input))
        return (handle_math(user_input))
    elif predicted_intent in response_dict:
        print(random.choice(response_dict[predicted_intent]))
        return (random.choice(response_dict[predicted_intent]))
    else:
        print("I'm not sure how to respond to that.")
        return ("I'm not sure how to respond to that.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def chat_response():
    # get user input
    user_msg = request.get_data().decode()
    print(user_msg)
    # use chat() to get chatbot's response
    chatbot_response = chat(user_msg)
    return jsonify({'response': chatbot_response})


if __name__ == "__main__":
    app.run(debug=True, port=5501)