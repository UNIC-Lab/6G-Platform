# -*- coding: utf-8 -*-

import json
import subprocess

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/generate_data", methods=["POST"])
def generate_data():
    try:
        print(111)
        data = request.get_json(force=True)
        print(222)
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
        
        with open("data.json", 'w', encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

        # [TODO] save data.json to database

        subprocess.run(["python", "main.py"])

        # [TODO] save files in ./data/ to database
        return jsonify({"message": "JSON data stored in file successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
