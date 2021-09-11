import random
import sys
from collections import deque, defaultdict
from pathlib import Path

from flask import Flask, request, render_template, redirect, url_for, current_app

from lmdb_loader import LmdbLoader

# alert: change to your username here after 'home'
app = Flask(__name__)
app.secret_key = 'super secret key kang'
app.config['UPLOAD_FOLDER'] = "static/"

files = defaultdict(deque)
all_info = defaultdict(deque)


@app.route('/index', methods=['POST', 'GET'])
@app.route('/', methods=['POST', 'GET'])
def index():
    if len(sys.argv) > 1:
        current_app.config["LMDB"] = sys.argv[1]
        return redirect(url_for('vis'))
    if request.method == 'POST':
        if 'lmdb_folder' in request.values:
            folder = request.values['lmdb_folder']
            if folder:
                current_app.config["LMDB"] = folder
                return redirect(url_for('vis'))
    return render_template('index.html')


@app.route('/vis', methods=['POST', 'GET'])
def vis():
    if "LMDB" not in current_app.config:
        return redirect(url_for("index"))
    folder = current_app.config["LMDB"]
    loader = LmdbLoader(folder)
    loader.load(folder)
    total_dataset = ", ".join(loader.dataset_names)
    count = loader.count()
    samples = []
    if count:
        for i in range(10):
            samples.append(list(loader.read(random.randint(0, count - 1))))
    Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
    for i, sample in enumerate(samples):
        image_bytes, label, name, dataset = sample
        img_path = f"{app.config['UPLOAD_FOLDER']}/{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(image_bytes)
        sample.append(f'{i}.jpg')

    version = random.randint(0, 100000000)
    return render_template('vis.html', count=loader.count_size, dataset=total_dataset, samples=samples, version=version)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)
