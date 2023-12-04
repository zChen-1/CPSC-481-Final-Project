from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import re
import ast
import operator
from sympy import symbols, Eq, solve, sympify

app = Flask(__name__)

# Load the training data
data = pd.read_csv('training_data/training_data.csv')
data_keyWord = pd.read_csv('training_data/training_data_keyWord.csv')

# Load the responses
response_data = pd.read_csv('training_data/responses.csv')

# Load the definitions
definitions = pd.read_csv('training_data/definition.csv')
definition_dict = dict(zip(definitions['Key Word'], definitions['Definition']))

# Create a dictionary
inputs = data['utterance']
labels = data['intent']
questions = data_keyWord['questions']  # Input from sentence
keyWords = data_keyWord['key word']  # Key Word from sentence

response_dict = {}
for intent in response_data['intent'].unique():
    response_dict[intent] = response_data[response_data['intent'] == intent]['response'].tolist()

# Split data into training and test sets
train_inputs, test_inputs, train_labels, test_labels = train_test_split(inputs, labels, test_size=0.2, random_state=42)
train_inputs_kw, test_inputs_kw, train_labels_kw, test_labels_kw = train_test_split(questions, keyWords, test_size=0.2,
                                                                                    random_state=42)

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model_kw = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model with the training data
model.fit(train_inputs, train_labels)
model_kw.fit(train_inputs_kw, train_labels_kw)


def safe_eval(node):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        left = safe_eval(node.left)
        right = safe_eval(node.right)
        if isinstance(node.op, ast.Add):  # add
            return operator.add(left, right)
        elif isinstance(node.op, ast.Sub):  # sub
            return operator.sub(left, right)
        elif isinstance(node.op, ast.Mult):  # multiply
            return operator.mul(left, right)
        elif isinstance(node.op, ast.Div):  # divide
            return operator.truediv(left, right)
        else:
            raise TypeError(node)
    else:
        raise TypeError(node)


# Function to handle math problems
# Limitation can only perform one type of operation
# e.g., 10 + 5 - 5 will return 20 (incorrect answer)
def handle_math(problem_statement):
    try:
        # Replace natural language terms with arithmetic symbols
        problem_statement = problem_statement.replace("add", "+").replace("plus", "+")
        problem_statement = problem_statement.replace("subtract", "-").replace("minus", "-")
        problem_statement = problem_statement.replace("multiply", "*").replace("times", "*")
        problem_statement = problem_statement.replace("divide", "/").replace("by", "/")

        node = ast.parse(problem_statement, mode='eval').body
        result = safe_eval(node)

        return f"The answer would be {result}."
    except Exception as e:
        return "There was a problem calculating that."


# Find the definition
def find_definition(query):
    words = query.lower().split()
    for word in words:
        if word in definition_dict:
            return definition_dict[word]
    return None


def is_calculus_query(query):
    calculus_keywords = ['derivative', 'integral', 'integrate', 'differentiate', 'sin', 'cos', 'tan']
    return any(keyword in query.lower() for keyword in calculus_keywords)


def handle_calculus(query):
    pass


# Function to handle algebraic equations
def is_algebraic_query(query):
    return re.search(r'\w+\s*[=]\s*.+', query)


def handle_algebra(query):
    try:
        # Replace common terms and isolate the equation
        query = query.replace("calculate", "").replace("what is", "").replace("=", "-(") + ")"
        # Define a symbol for x
        x = symbols('x')
        equation = sympify(query, locals={'x': x})
        result = solve(equation, x)  # output would be [number]
        if result:  # remove the "[" and "]" in output
            result_str = ', '.join(map(str, result))
            return f"X = {result_str}"
        else:
            return "I couldn't solve the equation."
    except Exception as e:
        return f"There was an error solving the equation: {e}"


# Chat, return response
def chat(user_input):
    # Check for math expression first
    if re.search(r'\b[a-z]+\b', user_input):  # Checks characters (like 'x'), but only work with 'x' now
        return handle_algebra(user_input)
    if re.search(r'\b\d+\s*(\+|\-|\*|\/|add|plus|subtract|minus|multiply|times|divide|by)\s*\d+\b', user_input,
                 re.IGNORECASE):
        return handle_math(user_input)
    if is_calculus_query(user_input):
        return handle_calculus(user_input)

    # Predict the intent
    predicted_intent = model.predict([user_input])[0]
    predicted_keyword = model_kw.predict([user_input])[0] if predicted_intent == 'definition' else None

    # Do not show to user, only for debugging
    print("Predicted Intent:", predicted_intent)
    if predicted_keyword:
        print("Key Word:", predicted_keyword)
    # Calculate the accuracy
    test_predictions = model.predict(test_inputs)
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Model Accuracy: {accuracy:.2%}")
    test_predictions_kw = model_kw.predict(test_inputs_kw)
    accuracy_kw = accuracy_score(test_labels_kw, test_predictions_kw)
    print(f"Model of find the key word Accuracy: {accuracy_kw:.2%}")

    # Handling different intents
    if predicted_intent == 'greeting':
        return random.choice(response_dict.get('greeting', ["Hello! How can I assist you?"]))
    elif predicted_intent == 'definition':
        return definition_dict.get(predicted_keyword, "I'm not sure how to respond to that.")
    elif predicted_intent in response_dict:
        return random.choice(response_dict[predicted_intent])
    else:
        return "I'm not sure how to respond to that."


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
