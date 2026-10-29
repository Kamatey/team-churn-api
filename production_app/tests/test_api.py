from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_invalid_prediction_is_rejected() -> None:
    response = client.post(
        "/predictions",
        json={
            "call_failure": -1,
        },
    )

    assert response.status_code == 422
