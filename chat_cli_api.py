from flask import Flask, request, jsonify
from chatbot import DocChatbot
import typer
from typing_extensions import Annotated

import glob


#VECTORDB_PATH = "./data/vector_store"
VECTORDB_PATH = "/opt/projects/openai/data/vector_store"
app = Flask(__name__)
docChatbot = DocChatbot()

@app.route("/ingest", methods=["POST"])
def ingest():
    path = request.form.get('path')
    name = request.form.get('name')

    file_list = glob.glob(path)
    docChatbot.init_vector_db_from_documents(file_list)
    docChatbot.save_vector_db_to_local(VECTORDB_PATH, name)

    return jsonify({"message": "Ingestion complete."})

@app.route("/chat", methods=["POST"])
def chat():
    #data = request.get_json()
    name = request.form.get('name')
    #the query is the question
    query = request.form.get('query')

    if query == "reset":
            chat_history = []
            return jsonify({"message": "Chat History successfully cleard."})
    
    docChatbot.load_vector_db_from_local(VECTORDB_PATH, name)
    docChatbot.init_chatchain()

    #chat_history = []

    result_answer, result_source = docChatbot.get_answer_with_source(query, chat_history)

    response = {
        "question": query,
        "answer": result_answer,
        "source_documents": [doc.metadata for doc in result_source]
    }

    #chat_history.append((query, result_answer))

    return jsonify(response)




if __name__ == "__main__":
	app.run(debug=True,use_reloader=False, port=8060)

