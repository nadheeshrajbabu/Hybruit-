from flask import Flask, request, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

interpreter = tf.lite.Interpreter(
    model_path=os.path.join(BASE_DIR, "model.tflite")
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(os.path.join(BASE_DIR, "labels.txt"), "r") as f:
    labels = [line.strip() for line in f.readlines()]

DISEASES = {
    "healthy tomato": "No disease detected by the model. Continue regular plant care.",
    "tomato early blight": "Possible early blight. Check for dark spots on older leaves. Remove affected leaves and avoid wetting foliage.",
    "healthy lime": "No disease detected by the model. Continue regular plant care.",
    "citrus canker": "Possible citrus canker. Look for raised, corky spots on leaves or fruit. Seek local agricultural advice.",
    "lime citrus greening": "Possible citrus greening. Look for blotchy yellow leaves and uneven fruit. Contact a local agricultural expert."
}

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hybruit</title>
<style>
body { font-family: Arial; background: #f0f7ed; margin: 0; padding: 20px; }
main { max-width: 500px; margin: 30px auto; background: white;
       padding: 24px; border-radius: 16px; text-align: center; }
h1 { color: #287a38; }
input, button { margin: 12px 0; max-width: 100%; }
button { background: #287a38; color: white; border: 0;
         padding: 12px 20px; border-radius: 8px; font-size: 16px; }
.result { background: #eef7eb; padding: 14px; border-radius: 10px; }
</style>
</head>
<body>
<main>
<h1>🌱 Hybruit</h1>
<p>Plant Disease Identification</p>
<form method="POST" enctype="multipart/form-data">
<input type="file" name="image" accept="image/*" required>
<br>
<button type="submit">Identify Plant</button>
</form>
{% if result %}
<div class="result">
<h2>{{ result }}</h2>
<p>Model confidence: {{ confidence }}%</p>
<p>{{ info }}</p>
</div>
{% endif %}
<p><small>AI predictions are estimates, not confirmed diagnoses.</small></p>
</main>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = confidence = info = None

    if request.method == "POST":
        uploaded = request.files.get("image")

        if not uploaded or not uploaded.filename:
            result = "Please select an image."
        else:
            try:
                image = Image.open(uploaded.stream).convert("RGB")
                height, width = input_details[0]["shape"][1:3]
                image = image.resize((width, height))

                data = np.asarray(image, dtype=np.float32)
                data = (data / 127.5) - 1.0
                data = np.expand_dims(data, axis=0)

                interpreter.set_tensor(input_details[0]["index"], data)
                interpreter.invoke()

                scores = interpreter.get_tensor(
                    output_details[0]["index"]
                )[0]

                index = int(np.argmax(scores))
                raw_label = labels[index]

                # Remove Teachable Machine's numeric label prefix.
                parts = raw_label.split(" ", 1)
                name = parts[1] if len(parts) == 2 and parts[0].isdigit() else raw_label
                result = name.strip().lower()

                if result == "class 3":
                    result = "unknown disease"

                confidence = round(float(scores[index]) * 100, 1)
                info = DISEASES.get(
                    result,
                    "The model cannot provide reliable disease information for this class. Please consult a plant specialist."
                )

                result = result.title()

            except Exception as e:
                result = "Could not process that image."
                info = "Try another clear plant-leaf image."

    return render_template_string(
        HTML, result=result, confidence=confidence, info=info
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
