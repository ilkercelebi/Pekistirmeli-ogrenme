"""Arayüzün kullandığı HTTP API için uçtan uca testler."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

import server.api as api  # noqa: E402
from binpick.scenarios import SCENARIOS, run_scenario  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "RESULTS", tmp_path)
    return TestClient(api.app)


def test_scenarios_have_intended_outcomes():
    expected = {"success": "success", "collision": "collision", "too_wide": "fail", "covered": "fail"}
    for sc in SCENARIOS:
        _, _, _, info = run_scenario(sc)
        assert info["outcome"] == expected[sc.key], sc.key


def test_meta_and_methods(client):
    m = client.get("/api/meta").json()
    assert m["n_states"] == 38880 and len(m["angles_deg"]) == 4
    assert client.get("/api/meta?gripper=vacuum").json()["angles_deg"] == [0.0]
    ids = [x["id"] for x in client.get("/api/methods").json()]
    assert ids == ["random", "heuristic"]


def test_session_step_and_preview(client):
    s = client.post("/api/session", json={"method": "heuristic", "seed": 3}).json()
    assert len(s["depth"]) == 44 and s["parts_left"] == len(s["parts"])
    r = client.post(f"/api/session/{s['id']}/step", json={}).json()
    ev = r["event"]
    assert ev["outcome"] in ("success", "collision", "fail")
    assert {"top_z", "stop_z"} <= set(ev["anim"])
    assert r["state"]["steps"] == 1 and len(r["parts_before"]) == len(s["parts"])

    # duvara çok yakın elle seçilmiş eylem çarpışmadır ve önizleme bunu bilir
    pv = client.post(f"/api/session/{s['id']}/preview", json={"angle_idx": 2, "row": 3, "col": 22}).json()
    assert pv["wall_hit"] is True
    r = client.post(f"/api/session/{s['id']}/step", json={"action": {"angle_idx": 2, "row": 3, "col": 22}}).json()
    assert r["event"]["outcome"] == "collision" and r["event"]["chosen_by"] == "user"

    maps = client.get(f"/api/session/{s['id']}/features?angle_idx=1").json()
    assert {"yükseklik", "duvar", "genişlik"} <= set(maps)


def test_unknown_session_is_404(client):
    assert client.post("/api/session/yok/step", json={}).status_code == 404


def test_evaluate(client):
    r = client.post("/api/evaluate", json={"methods": ["random", "heuristic"], "episodes": 3}).json()
    assert [x["method"] for x in r["results"]] == ["random", "heuristic"]


def test_training_job_produces_usable_run(client):
    job = client.post("/api/train", json={"episodes": 12, "name": "t"}).json()
    for _ in range(300):
        st = client.get(f"/api/train/{job['id']}").json()
        if st["status"] != "running":
            break
        time.sleep(0.1)
    assert st["status"] == "done", st
    assert "t" in [x["id"] for x in client.get("/api/methods").json()]
    detail = client.get("/api/runs/t?window=5").json()
    assert detail["visited"] > 0 and "duvar" in detail["marginal"]
    s = client.post("/api/session", json={"method": "t", "seed": 1}).json()
    assert s["learns"] and s["qmap"] is not None
