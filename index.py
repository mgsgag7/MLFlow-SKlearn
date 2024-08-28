from flask import Flask, render_template
from train import best_fit_models

app = Flask(__name__)

@app.route('/train')
def train():
    best_fit_models()
    return render_template("run_model.html")

@app.route('/')
def index():
    return render_template("index.html")

# main driver function
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)