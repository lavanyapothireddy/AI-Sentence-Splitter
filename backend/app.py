from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=False)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are an expert AI Sentence Splitter. Your job is to intelligently split a given text into individual, clean, and meaningful sentences.

Rules:
1. Split the text into individual sentences.
2. Each sentence must be complete and meaningful.
3. Preserve the original wording — do NOT rephrase or modify.
4. Handle abbreviations (e.g., Mr., Dr., U.S.A.) correctly — do NOT split them.
5. Handle ellipsis (...) carefully — only split if it ends a complete sentence.
6. Return ONLY a valid JSON array of sentences. Example: ["Sentence one.", "Sentence two."]
7. Do NOT include any explanation, markdown, or extra text — only the JSON array."""

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "AI Sentence Splitter API is running!"})

@app.route("/split", methods=["POST"])
def split_sentences():
    data = request.get_json()
    
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in request body."}), 400
    
    text = data["text"].strip()
    
    if not text:
        return jsonify({"error": "Text cannot be empty."}), 400
    
    if len(text) > 5000:
        return jsonify({"error": "Text too long. Maximum 5000 characters allowed."}), 400

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Split this text into sentences:\n\n{text}"}
            ],
            model="llama3-8b-8192",
            temperature=0.1,
            max_tokens=2048,
        )
        
        raw_response = chat_completion.choices[0].message.content.strip()

        import json
        # Clean up if model wraps in markdown
        if raw_response.startswith("```"):
            raw_response = raw_response.strip("`").strip()
            if raw_response.startswith("json"):
                raw_response = raw_response[4:].strip()

        sentences = json.loads(raw_response)
        
        if not isinstance(sentences, list):
            raise ValueError("Response is not a list")

        return jsonify({
            "success": True,
            "original_text": text,
            "sentences": sentences,
            "sentence_count": len(sentences)
        })

    except json.JSONDecodeError:
        return jsonify({"error": "Failed to parse AI response. Please try again."}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
