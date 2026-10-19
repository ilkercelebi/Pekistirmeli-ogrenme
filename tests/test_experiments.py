"""run_experiments.py uçtan uca: küçük bir deney eğitilir, değerlendirilir, rapor ve README bloğu üretilir."""

import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_experiments as rx  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_end_to_end(tmp_path):
    readme = tmp_path / "README.md"
    shutil.copy(ROOT / "README.md", readme)
    out, img = tmp_path / "results", tmp_path / "img"
    args = ["--episodes", "8", "--seeds", "0", "1", "--algos", "q_learning", "--grippers", "parallel_jaw",
            "--test-boxes", "2", "--eval-episodes", "2", "--workers", "1", "--no-gifs",
            "--out", str(out), "--readme", str(readme), "--img-dir", str(img)]
    rx.main(args)

    for s in (0, 1):
        assert rx.is_complete(out / f"q_learning_parallel_jaw_s{s}", 8)
    agg = json.loads((out / "report" / "summary_parallel_jaw.json").read_text())
    assert set(agg) == {"random", "heuristic", "oracle", "q_learning"}
    assert agg["q_learning"]["n_seeds"] == 2 and agg["q_learning"]["success_rate_seed_std"] is not None
    for name in ("learning_curves_parallel_jaw.png", "greedy_parallel_jaw.png", "bars_parallel_jaw.png", "qtable_parallel_jaw.png"):
        assert (img / name).exists(), name

    text = readme.read_text()
    assert re.search(r"\| (\*\*)?Q-learning(\*\*)? \(2 tohum\) \|", text) and "otomatik dolar" not in text
    assert "Kâhin (üst sınır)" in text
    assert "learning_curves_parallel_jaw.png" in text

    # ikinci çalıştırma: eğitimler atlanır, README blokları yine tek kopya kalır
    rx.main(args)
    text = readme.read_text()
    assert text.count("<!-- SONUCLAR:BASLA") == 1 and text.count("**Ana bulgu:**") == 1


def test_vacuum_ablation_compares_information_and_learning(tmp_path):
    readme = tmp_path / "README.md"
    shutil.copy(ROOT / "README.md", readme)
    out, img = tmp_path / "results", tmp_path / "img"
    rx.main(["--episodes", "4", "--seeds", "0", "1", "--algos", "q_learning", "--grippers", "vacuum", "--ablation",
             "--test-boxes", "1", "--eval-episodes", "1", "--workers", "1", "--no-gifs",
             "--out", str(out), "--readme", str(readme), "--img-dir", str(img)])

    # yalnız vakum satırı çalışır, ana deneyle aynı tohumlarla; parmaklı ablasyonu atlanır
    assert sorted(p.name for p in (out / "ablation").iterdir()) == ["vacuum_noface_s0", "vacuum_noface_s1", "vacuum_nomem", "vacuum_nostop", "vacuum_nostreak_s0", "vacuum_nostreak_s1"]
    cfg = json.loads((out / "ablation" / "vacuum_noface_s0" / "config.json").read_text())
    assert cfg["features"]["face_feature"] is False
    agg = json.loads((out / "report" / "summary_vacuum.json").read_text())
    assert {"heuristic", "heuristic_face", "q_learning"} <= set(agg)
    assert (img / "ablation_vacuum.png").exists() and not (img / "ablation.png").exists()

    text = readme.read_text()
    assert "| Sezgisel + yüzey bilgisi | var | — |" in text
    assert "| Q-learning (2 tohum) | yok | 1728 hücre |" in text and "| Q-learning (2 tohum) | var | 3456 hücre |" in text
    assert "Yüzey bilgisinin etkisi" in text
