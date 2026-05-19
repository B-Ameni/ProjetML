import time
import requests


def test_mlflow_serve_ping():
    import requests

    response = requests.get("http://127.0.0.1:1234/ping")

    assert response.status_code == 200
if __name__ == '__main__':
    test_mlflow_serve_ping()
    print('Test API terminé avec succès.')
