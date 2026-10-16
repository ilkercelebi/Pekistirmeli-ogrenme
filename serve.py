"""Arayüzü başlatır: API + derlenmiş React uygulaması tek adreste.

    python serve.py             → http://localhost:8000

Arayüz geliştirme modunda (canlı yenileme) çalıştırmak için bkz. web/README.md.
"""

import argparse

import uvicorn

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args()
    print(f"Arayüz: http://localhost:{a.port}")
    uvicorn.run("server.api:app", host=a.host, port=a.port)
