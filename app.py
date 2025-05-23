import numpy as np
import os
import tensorflow as tf
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app  = Flask(__name__)
MODEL_DIR = os.getenv('SERVING_MODEL_DIR', 'wildan14ar-pipeline/serving_model/1748013612')
model = tf.saved_model.load(MODEL_DIR)
infer = model.signatures['serving_default']

PREDICTION_COUNT = Counter('prediction_requests_total', 'Total number of prediction requests')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Latency for prediction requests')

@app.route('/predict', methods=['POST'])
def predict():
    PREDICTION_COUNT.inc()

    # 1. Parsing JSON (dict {'instances': [...] } or list of {'text':...})
    payload = request.get_json(force=True)
    if isinstance(payload, dict):
        instances = payload.get('instances', [])
    elif isinstance(payload, list):
        instances = payload
    else:
        return jsonify({"error": "Unsupported payload format"}), 400
    if not instances:
        return jsonify({"error": "'instances' is empty"}), 400

    texts = []
    for i, inst in enumerate(instances):
        if not isinstance(inst, dict) or 'text' not in inst:
            return jsonify({"error": f"Instance at index {i} missing 'text'"}), 400
        texts.append(inst['text'])

    # 2. Serialize each text ke tf.train.Example dengan feature 'message'
    serialized = []
    for txt in texts:
        ex = tf.train.Example(features=tf.train.Features(feature={
            'message': tf.train.Feature(bytes_list=tf.train.BytesList(
                          value=[txt.encode('utf-8')]))
        }))
        serialized.append(ex.SerializeToString())

    input_tensor = tf.constant(serialized)

    # 3. Panggil signature dengan keyword sesuai nama input: 'examples'
    with PREDICTION_LATENCY.time():
        preds = infer(examples=input_tensor)

    # 4. Ambil output probabilitas, flatten, threshold 0.5
    probs = preds['output_0'].numpy().flatten()
    labels = (probs > 0.5).astype(int).tolist()

    return jsonify({
      'predictions': labels,
      'probabilities': probs.tolist()
    })

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/healthz')
def healthz():
    return 'OK', 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
