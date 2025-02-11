from flask import Flask, request, jsonify, send_file
import os
import openai
import subprocess
import json
import datetime
import sqlite3
import re
import glob
import requests
import shutil
import duckdb
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer, util
import markdown
import csv
import pandas as pd

def safe_path(path):
    """ Ensure the path is within /data """
    abs_path = os.path.abspath(path)
    if not abs_path.startswith("/data"):
        raise PermissionError("Access outside /data is not allowed")
    return abs_path

def execute_task(task_description):
    """ Parses and executes a given task using an LLM where needed. """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": task_description}],
            api_key=os.environ["AIPROXY_TOKEN"]
        )

        command = response["choices"][0]["message"]["content"]
        return process_command(command)
    except openai.error.OpenAIError as e:
        return str(e), 400
    except Exception as e:
        return str(e), 500

def process_command(command):
    try:
        if "install uv" in command:
            subprocess.run(["pip", "install", "uv"], check=True)
            subprocess.run(["python", "datagen.py", os.getenv("USER_EMAIL")], check=True)
            return "Data generation completed.", 200
        
        elif "format markdown" in command:
            subprocess.run(["npx", "prettier@3.4.2", "--write", safe_path("/data/format.md")], check=True)
            return "Markdown formatted.", 200
        
        elif "fetch data from API" in command:
            url = command.split("fetch data from API ")[1].strip()
            response = requests.get(url)
            with open(safe_path("/data/api-data.json"), "w") as f:
                f.write(response.text)
            return "API data saved.", 200
        
        elif "clone git repo" in command:
            repo_url = command.split("clone git repo ")[1].strip()
            subprocess.run(["git", "clone", repo_url, safe_path("/data/repo")], check=True)
            return "Git repo cloned.", 200
        
        elif "resize image" in command:
            img_path = safe_path("/data/image.jpg")
            image = Image.open(img_path)
            image = image.resize((800, 600))
            image.save(safe_path("/data/image_resized.jpg"))
            return "Image resized.", 200
        
        elif "transcribe audio" in command:
            audio_path = safe_path("/data/audio.mp3")
            transcript = subprocess.run(["whisper", audio_path], capture_output=True, text=True).stdout
            with open(safe_path("/data/audio-transcript.txt"), "w") as f:
                f.write(transcript)
            return "Audio transcribed.", 200
        
        elif "convert markdown to html" in command:
            with open(safe_path("/data/content.md")) as f:
                html_content = markdown.markdown(f.read())
            with open(safe_path("/data/content.html"), "w") as f:
                f.write(html_content)
            return "Markdown converted to HTML.", 200
        
        elif "run sql query" in command:
            db_path = safe_path("/data/database.db")
            query = command.split("run sql query ")[1].strip()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            conn.close()
            with open(safe_path("/data/sql-result.json"), "w") as f:
                json.dump(result, f, indent=2)
            return "SQL query executed.", 200
        
        elif "filter CSV" in command:
            df = pd.read_csv(safe_path("/data/data.csv"))
            filtered_df = df[df["status"] == "active"]
            result = filtered_df.to_json(orient="records")
            with open(safe_path("/data/data-filtered.json"), "w") as f:
                f.write(result)
            return "Filtered CSV data.", 200
        
        else:
            return "Unknown command.", 400
    except Exception as e:
        return str(e), 500

@app.route('/run', methods=['POST'])
def run_task():
    task = request.args.get('task')
    if not task:
        return jsonify({"error": "Missing task description"}), 400
    
    result, status_code = execute_task(task)
    return jsonify({"result": result}), status_code

@app.route('/read', methods=['GET'])
def read_file():
    file_path = request.args.get('path')
    if not file_path or not os.path.exists(safe_path(file_path)):
        return "", 404
    
    return send_file(file_path, as_attachment=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)