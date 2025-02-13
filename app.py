from flask import Flask, request, jsonify, send_file
from bs4 import BeautifulSoup
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

app = Flask(__name__)  # Ensure this is present


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
        
        elif "count Wednesdays" in command:
            with open(safe_path("/data/dates.txt")) as f:
                dates = [line.strip() for line in f.readlines()]
            wednesday_count = sum(1 for date in dates if datetime.datetime.strptime(date, "%Y-%m-%d").weekday() == 2)
            with open(safe_path("/data/dates-wednesdays.txt"), "w") as f:
                f.write(str(wednesday_count))
            return "Counted Wednesdays.", 200

        elif "sort contacts" in command:
            with open(safe_path("/data/contacts.json")) as f:
                contacts = json.load(f)
            sorted_contacts = sorted(contacts, key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))
            with open(safe_path("/data/contacts-sorted.json"), "w") as f:
                json.dump(sorted_contacts, f, indent=2)
            return "Sorted contacts.", 200

        elif "extract recent logs" in command:
            log_files = sorted(glob.glob(safe_path("/data/logs/*.log")), key=os.path.getmtime, reverse=True)[:10]
            with open(safe_path("/data/logs-recent.txt"), "w") as f:
                for log_file in log_files:
                    with open(log_file) as log:
                        first_line = log.readline().strip()
                        f.write(first_line + "\n")
            return "Extracted recent logs.", 200

        elif "index markdown files" in command:
            index = {}
            for md_file in glob.glob(safe_path("/data/docs/*.md")):
                with open(md_file) as f:
                    for line in f:
                        if line.startswith("# "):  
                            index[os.path.basename(md_file)] = line[2:].strip()
                            break  
            with open(safe_path("/data/docs/index.json"), "w") as f:
                json.dump(index, f, indent=2)
            return "Indexed markdown files.", 200

        elif "extract email sender" in command:
            with open(safe_path("/data/email.txt")) as f:
                email_content = f.read()
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Extract sender email from:\n{email_content}"}],
                api_key=os.environ["AIPROXY_TOKEN"]
            )
            sender_email = response["choices"][0]["message"]["content"].strip()
            with open(safe_path("/data/email-sender.txt"), "w") as f:
                f.write(sender_email)
            return "Extracted email sender.", 200

        elif "extract credit card number" in command:
            image = Image.open(safe_path("/data/credit-card.png"))
            card_number = pytesseract.image_to_string(image).replace(" ", "").strip()
            with open(safe_path("/data/credit-card.txt"), "w") as f:
                f.write(card_number)
            return "Extracted credit card number.", 200

        elif "find similar comments" in command:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            with open(safe_path("/data/comments.txt")) as f:
                comments = [line.strip() for line in f.readlines()]
            embeddings = model.encode(comments, convert_to_tensor=True)
            similarity_scores = util.pytorch_cos_sim(embeddings, embeddings)
            max_score, best_pair = -1, None
            for i in range(len(comments)):
                for j in range(i + 1, len(comments)):
                    if similarity_scores[i][j] > max_score:
                        max_score = similarity_scores[i][j]
                        best_pair = (comments[i], comments[j])
            with open(safe_path("/data/comments-similar.txt"), "w") as f:
                f.write("\n".join(best_pair))
            return "Identified similar comments.", 200

        elif "compute gold ticket sales" in command:
            conn = sqlite3.connect(safe_path("/data/ticket-sales.db"))
            cursor = conn.cursor()
            cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
            total_sales = cursor.fetchone()[0] or 0
            conn.close()
            with open(safe_path("/data/ticket-sales-gold.txt"), "w") as f:
                f.write(str(total_sales))
            return "Computed Gold ticket sales.", 200



        elif "fetch data from API" in command:
            url = command.split("fetch data from API ")[1].strip()
            response = requests.get(url)
            with open(safe_path("/data/api-data.json"), "w") as f:
                f.write(response.text)
            return "API data saved.", 200
        
        elif "clone git repo" in command:
            repo_url = command.split("clone git repo ")[1].strip()
            repo_path = safe_path("/data/repo")
            subprocess.run(["git", "clone", repo_url, repo_path], check=True)
            with open(os.path.join(repo_path, "update.txt"), "w") as f:
                f.write("Automated update\n")
            subprocess.run(["git", "-C", repo_path, "add", "."], check=True)
            subprocess.run(["git", "-C", repo_path, "commit", "-m", "Automated commit"], check=True)
            subprocess.run(["git", "-C", repo_path, "push"], check=True)  # Ensure the repo allows pushing

            return "Git repo cloned.", 200
            
        
        elif "run duckdb query" in command:
            db_path = safe_path("/data/database.duckdb")
            query = command.split("run duckdb query ")[1].strip()
            con = duckdb.connect(db_path)
            result = con.execute(query).fetchall()
            con.close()
            with open(safe_path("/data/duckdb-result.json"), "w") as f:
                json.dump(result, f, indent=2)
            return "DuckDB query executed.", 200


        elif "resize image" in command:
            img_path = safe_path("/data/image.jpg")
            image = Image.open(img_path)
            image = image.resize((800, 600))
            image.save(safe_path("/data/image_resized.jpg"))
            return "Image resized.", 200

        elif "compress image" in command:
            img_path = safe_path("/data/image.jpg")
            image = Image.open(img_path)
            image.save(safe_path("/data/image_compressed.jpg"), "JPEG", quality=50)
            return "Image compressed.", 200

        
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

        elif "scrape website" in command:
            url = command.split("scrape website ")[1].strip()
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            text_data = soup.get_text()  # Extract readable text
            with open(safe_path("/data/scraped-content.txt"), "w") as f:
                f.write(text_data)
            return "Website scraped.", 200

        
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

