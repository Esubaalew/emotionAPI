from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Rate limiting configuration
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=7)
model.load_state_dict(torch.load('emotion_model.pth', map_location=torch.device('cpu')))
model.eval()

with open('emotion_tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🎭 Emotion Detection AI</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            margin: 0;
            padding: 20px;
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 25px;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
            min-height: 150px;
            transition: border-color 0.3s ease;
        }
        textarea:focus {
            border-color: #3498db;
            outline: none;
        }
        button {
            background: #3498db;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s ease, background 0.3s ease;
            display: block;
            margin: 0 auto;
        }
        button:hover {
            background: #2980b9;
            transform: translateY(-2px);
        }
        .result {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            border-radius: 8px;
            background: #f8f9fa;
            display: none;
            animation: fadeIn 0.5s ease;
        }
        .spinner {
            display: none;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .emotion-display {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 AI Emotion Detector</h1>
        <div class="form-group">
            <textarea id="text-input" placeholder="Enter your text here..."></textarea>
        </div>
        <button onclick="analyzeEmotion()">Analyze Emotion</button>
        <div class="spinner" id="spinner"></div>
        <div class="result" id="result">
            <div class="emotion-display" id="emotion-result"></div>
        </div>
    </div>

    <script>
        async function analyzeEmotion() {
            const text = document.getElementById('text-input').value;
            const resultDiv = document.getElementById('result');
            const spinner = document.getElementById('spinner');
            const emotionResult = document.getElementById('emotion-result');

            if (!text) {
                alert('Please enter some text to analyze!');
                return;
            }

            spinner.style.display = 'block';
            resultDiv.style.display = 'none';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });

                if (!response.ok) throw new Error('Analysis failed');

                const result = await response.text();
                const parser = new DOMParser();
                const doc = parser.parseFromString(result, 'text/html');
                const emotion = doc.querySelector('h2').textContent.split(': ')[1];

                emotionResult.textContent = `Detected Emotion: ${emotion}`;
                resultDiv.style.backgroundColor = getEmotionColor(emotion);
                resultDiv.style.display = 'block';
            } catch (error) {
                emotionResult.textContent = 'Error analyzing text. Please try again.';
                resultDiv.style.display = 'block';
            } finally {
                spinner.style.display = 'none';
            }
        }

        function getEmotionColor(emotion) {
            const colors = {
                anger: '#ff4757',
                disgust: '#2ed573',
                fear: '#57606f',
                joy: '#ffa502',
                neutral: '#dfe4ea',
                sadness: '#1e90ff',
                surprise: '#ff6348'
            };
            return colors[emotion.toLowerCase()] || '#f8f9fa';
        }
    </script>
</body>
</html>
"""


def predict_emotion(text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    return torch.argmax(logits, dim=1).item()


@app.route('/')
@limiter.limit("10/minute")
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
@limiter.limit("5/minute")
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing text parameter"}), 400

    try:
        emotion_class = predict_emotion(data['text'])
        predicted_emotion = emotion_labels[emotion_class]
        return f"<h2>Predicted Emotion: {predicted_emotion}</h2>", 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "message": "Emotion Detection API is running!"})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"message": "Endpoint not found"}), 404


@app.errorhandler(429)
def rate_limit_exceeded(e):
    return jsonify({"error": "Rate limit exceeded"}), 429


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)