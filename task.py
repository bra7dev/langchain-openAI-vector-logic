import psycopg2
import json
import numpy as np
# import faiss
import openai
import re
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from sqlalchemy import text
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import (
    create_engine,
    MetaData,
    inspect,
    Table,
    Column,
    Integer,
    String,
    Boolean,
    ARRAY,
    Float,
    Sequence,
    JSON,
)
from sqlalchemy.exc import SQLAlchemyError
# from database_manager import DatabaseManager
from flask_jwt_extended import (
    jwt_required,
    verify_jwt_in_request,
    get_jwt_identity,
    JWTManager,
)
from flask_cors import CORS, cross_origin
from functools import wraps
import datetime
import logging
import time

# Setup logging
# logging.basicConfig(filename='logs/page_similarity_v4.4.log', level=logging.DEBUG, format='%(asctime)s [%(levelname)s] - %(message)s')
# logger = logging.getLogger()


def jwt_required_except_options(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if request.method != "OPTIONS":
            try:
                verify_jwt_in_request()
            except Exception as e:
                logging.error(f"Error when verifying JWT token: {e}")
                return (
                    jsonify({"message": "Error when verifying JWT token"}),
                    401,
                )  # Unauthorized
        return fn(*args, **kwargs)

    return wrapper


app = Flask(__name__)
jwt = JWTManager(app)

app.config["JWT_SECRET_KEY"] = "1234"  # Change this to the actual JWT secret key
app.config["JWT_HEADER_NAME"] = "Authorization"
app.config["JWT_HEADER_TYPE"] = "Bearer"
app.config["JWT_TOKEN_LOCATION"] = ["headers"]
# app.debug = True

CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Add OpenAI configurations
openai.api_key = "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"
model_engine = "gpt-4"


def chat_with_ai(user_input):
    conversation_history = [
        {
            "role": "system",
            "content": "You are an AI assistant that provides helpful answers.",
        },
        {"role": "user", "content": user_input},
    ]

    response = openai.ChatCompletion.create(
        model=model_engine,
        messages=conversation_history,
        max_tokens=2000,
        n=1,
        stop=None,
        temperature=0.7,
    )

    ai_response = response.choices[0].message["content"]
    print("OpenAI API Response:", response)
    app.logger.info("OpenAI API Response: %s", response)

    return ai_response

def insert_db(user_id, query, features_info):
    #  print("database", feature)
    try:
        db = psycopg2.connect(
            host="internaldb.c0rz9kyyn4jp.us-east-2.rds.amazonaws.com",
            user="postgres",
            password="Kwantx11!!",
            database="internaldb"
        )
        print("PostgreSQL connection is succssful!")
    except (psycopg2.Error, Exception) as error:
        print(f"DB connection is error, {error}")

    cursor = db.cursor()    
    db.set_session(autocommit=True)

    for feature in features_info:
        try:
            feature_name, feature_formula, feature_definition, feature_parameters = feature
            sql = f"INSERT INTO feature_from_query (user_id, query, feature_name, feature_definition, feature_formula, feature_parameters) VALUES ('{user_id}', '{query}', '{feature_name}', '{feature_definition}', '{feature_formula}', '{feature_parameters}')"
            cursor.execute(sql)
            print("db state", sql)
        except:
            # print("Error occured!", error)
            continue
    cursor.close()
    db.close()


def Parse_data(data):
    print(data)
    pattern = re.compile(r'\d+A\) Feature Name: (.*?)\n'
                            r'\d+B\) Feature Formula: (.*?)\n'
                            r'\d+C\) Feature Definition: (.*?)\n'
                            r'\d+D\) Feature Parameters: (.*?)\n\n', re.DOTALL)
    result = pattern.findall(str(data))

    return result

@app.route("/handle_query", methods=["POST"])
# @cross_origin()
# @jwt_required_except_options
def handle_query():
    app.logger.info("Processing /selected_papers request...")
    start_time = datetime.datetime.now()

    data = request.json
    print("Inside /selected_papers route...")
    # logger.info("Inside /selected_papers route...")
    print(f"Data received: {data}")
    # logger.info("Data received: %s", data)

    user_id, query = data.get('user_id'), data.get('query')

    if None in (user_id, query):
        return (
            jsonify({"message": "Error: Missing required parameters."}),
            400,
        )  # Bad Request

    print(
        f"User ID: {user_id}, Query: {query}"
    )
    # logger.info(
    #     "User ID: %s, Query: %s, Selected Paper IDs: %s",
    #     user_id,
    #     query,
    # )

    k = 10  # number of closest pages to return

    

    prompt = f"""
    As an AI, I should understand question and provide information about 11 features in a specific format. For each feature, I need to provide feature in following format

    0A) Feature Name: The name of the feature
    0B) Feature Formula: The mathematical formula for the feature
    0C) Feature Definition: A brief description or definition of the feature
    0D) Feature Parameters: Parameters or variables used in the feature formula as json format, pls provide json format

    and number each one incrementally (start from 0 to 10. so feature#1 is 0A, 0B...feature #2 is 1A, 1B..feature #3 is 2A, feature #4 is 3A, feature #5 is 4A... feature #10 is 9A, feature #11 is 10A

    This is my question:
    {query}

    Please provide the 11 feature information according to the above format.
    """

    ai_chat = chat_with_ai(prompt)

    # print(ai_chat)

    data = f"""{ai_chat}"""

    # print(data)
    features_info = Parse_data(data)

    # features_info = pattern.findall(ai_chat)
    
    print("features-info***************", features_info)

    insert_db(user_id, query, features_info)
    features = []
    for feature in features_info:
        try:
            feature_name, feature_formula, feature_definition, feature_parameters = feature 
            
        except:
            continue
        # logger.info(f'feature_name: {feature_name}, feature_formula: {feature_formula},  feature_definition: {feature_definition}, feature_parameters: {feature_parameters}')

        if not all([feature_name, feature_formula, feature_definition, feature_parameters]):
            continue

        try:
            decoded_parameters = json.loads(feature_parameters)
            final_feature_parameters = decoded_parameters
        except json.JSONDecodeError:
            final_feature_parameters = {}

        # Prepare a feature dictionary
        featureObj = {
            "user_id": user_id,
            "feature_name": feature_name,
            "feature_definition": feature_definition,
            "feature_formula": feature_formula,
            "feature_parameters": final_feature_parameters
        }

        # logger.info(f' ====== feature:  {feature}')

        features.append(featureObj)

    return ( jsonify({"data": features}), 200 )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010)
