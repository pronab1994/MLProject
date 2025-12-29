import time
from flask import Flask, request, render_template, jsonify

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# ============================================================
# Elastic Beanstalk WSGI entrypoint
# ============================================================
# EB expects a callable named `application`
application = Flask(__name__)
app = application


# ============================================================
# Lazy-load ML pipeline (CRITICAL for EB)
# ============================================================
# Loading models at import time causes Gunicorn to crash on EB
# This guarantees the server starts first (no 502)
_predict_pipeline = None

def get_pipeline():
    global _predict_pipeline
    if _predict_pipeline is None:
        _predict_pipeline = PredictPipeline()
    return _predict_pipeline


# ============================================================
# Request timing (optional logging)
# ============================================================
@app.before_request
def _start_timer():
    request._start_time = time.perf_counter()


@app.after_request
def _log_request_time(response):
    try:
        elapsed_ms = (time.perf_counter() - request._start_time) * 1000
        print(
            f"{request.method} {request.path} "
            f"-> {response.status_code} in {elapsed_ms:.2f} ms"
        )
    except Exception:
        pass
    return response


# ============================================================
# Health check (USED BY EB / DEBUGGING)
# ============================================================
@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/ping")
def ping():
    return "pong"


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html", results=None, error=None)

    try:
        required_fields = [
            "gender",
            "ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
            "reading_score",
            "writing_score",
        ]

        missing = [f for f in required_fields if not request.form.get(f)]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        reading_score = float(request.form.get("reading_score"))
        writing_score = float(request.form.get("writing_score"))

        if not (0 <= reading_score <= 100):
            raise ValueError("Reading score must be between 0 and 100.")
        if not (0 <= writing_score <= 100):
            raise ValueError("Writing score must be between 0 and 100.")

        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=reading_score,
            writing_score=writing_score,
        )

        pred_df = data.get_data_as_data_frame()

        preds = get_pipeline().predict(pred_df)

        result = float(preds[0])
        result = max(0.0, min(100.0, result))

        return render_template("home.html", results=result, error=None)

    except Exception as e:
        return render_template("home.html", results=None, error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
