"""Check real HTTP services after docker compose up --wait (stdlib only)."""
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    with urlopen("http://127.0.0.1:8000/health", timeout=20) as response:
        assert json.load(response)["status"] == "ok"
    with urlopen("http://127.0.0.1:8501/_stcore/health", timeout=20) as response:
        assert response.status == 200
    request = Request("http://127.0.0.1:8000/predict",
                      data=(ROOT / "examples/predict_request.json").read_bytes(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
    expected = json.loads((ROOT / "examples/predict_response.json").read_text(encoding="utf-8"))
    assert abs(result["score"] - expected["score"]) < 1e-10
    assert result["model_version"] == expected["model_version"]
    try:
        with urlopen(Request("http://127.0.0.1:8000/predict", data=b'{}',
                             headers={"Content-Type": "application/json"}), timeout=20):
            raise AssertionError("Invalid request unexpectedly succeeded")
    except HTTPError as error:
        assert error.code == 422
    print(json.dumps({"api": "ok", "dashboard": "ok", "invalid_request": 422,
                      "score": result["score"], "model_version": result["model_version"]}))


if __name__ == "__main__":
    main()
