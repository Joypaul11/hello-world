import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, flash, redirect, render_template, request, url_for


logging.basicConfig(level=logging.INFO)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit by default
    app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me")

    detector = HuggingFaceDetector.from_env()

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            file = request.files.get("image")

            if not file or file.filename == "":
                flash("Please choose an image file to upload.")
                return redirect(url_for("index"))

            try:
                image_bytes = file.read()
                if not image_bytes:
                    flash("Uploaded file is empty.")
                    return redirect(url_for("index"))

                result = detector.analyze(image_bytes)

                preview = base64.b64encode(image_bytes).decode("utf-8")
                mime_type = file.mimetype or "image/jpeg"

                return render_template(
                    "index.html",
                    result=result,
                    preview_src=f"data:{mime_type};base64,{preview}",
                )
            except DetectorConfigurationError as exc:
                logging.exception("Detector misconfiguration")
                flash(str(exc))
                return redirect(url_for("index"))
            except DetectorServiceError as exc:
                logging.exception("External detection service error")
                flash(
                    "The detection service returned an error. Please try again later or "
                    "check the server logs for details."
                )
                return render_template("index.html", service_error=str(exc))
            except Exception:
                logging.exception("Unexpected error while analyzing image")
                flash("An unexpected error occurred while analyzing the image.")
                return redirect(url_for("index"))

        return render_template("index.html")

    return app


class DetectorConfigurationError(RuntimeError):
    """Raised when detector configuration is invalid."""


class DetectorServiceError(RuntimeError):
    """Raised when the external detection service fails."""


@dataclass
class DetectionResult:
    label: str
    score: float
    raw: Any

    @property
    def confidence_percent(self) -> str:
        return f"{self.score * 100:.1f}%"


class HuggingFaceDetector:
    def __init__(self, api_url: str, token: Optional[str] = None, timeout: float = 15.0) -> None:
        self.api_url = api_url
        self.token = token
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "HuggingFaceDetector":
        model_id = os.getenv("HF_MODEL_ID", "orionw/ai-image-detection")
        base_url = os.getenv("HF_API_URL") or f"https://api-inference.huggingface.co/models/{model_id}"
        token = os.getenv("HF_TOKEN")

        if not base_url:
            raise DetectorConfigurationError(
                "Hugging Face API URL is not configured. Set HF_API_URL or HF_MODEL_ID."
            )

        return cls(api_url=base_url, token=token)

    def analyze(self, image_bytes: bytes) -> DetectionResult:
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        response = requests.post(
            self.api_url,
            headers=headers,
            data=image_bytes,
            timeout=self.timeout,
        )

        if response.status_code == 503:
            raise DetectorServiceError(
                "Model is loading on Hugging Face. Please retry after a few seconds."
            )

        if response.status_code >= 400:
            raise DetectorServiceError(
                f"Hugging Face API error {response.status_code}: {response.text[:200]}"
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise DetectorServiceError("Invalid JSON response from Hugging Face API") from exc

        top_result = self._parse_top_result(payload)
        logging.info("Detection result: %s", top_result)
        return DetectionResult(label=top_result["label"], score=top_result["score"], raw=payload)

    @staticmethod
    def _parse_top_result(payload: Any) -> Dict[str, Any]:
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, list) and first:
                first = first[0]
            if "label" in first:
                return {"label": first["label"], "score": float(first.get("score", 0.0))}

        if isinstance(payload, dict) and "label" in payload:
            return {"label": payload["label"], "score": float(payload.get("score", 0.0))}

        raise DetectorServiceError("Unexpected response format from Hugging Face API")


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
