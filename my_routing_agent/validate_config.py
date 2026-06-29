"""Pre-flight configuration validation for ANGKOR + PHANTOM."""

from __future__ import annotations

import sys

import requests

from my_routing_agent.config import load_config


def check_ollama() -> bool:
    config = load_config()
    base = config.local.base_url
    if not base:
        base = "http://localhost:11434/v1"
    server_base = base.rstrip("/v1").rstrip("/")
    url = f"{server_base}/api/tags"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            names = {m.get("name", "") for m in models}
            local_model = config.local.model
            if local_model in names or f"{local_model}:latest" in names:
                print(f"✅ Ollama reachable — model '{local_model}' found")
            else:
                print(f"⚠️  Ollama reachable, but model '{local_model}' not pulled ({len(names)} models available)")
            return True
        print(f"❌ Ollama returned status {resp.status_code}")
        return False
    except requests.ConnectionError:
        print("❌ Ollama is NOT running on localhost:11434. Start with: ollama serve")
        return False


def check_fireworks() -> bool:
    config = load_config()
    if not config.remote.api_key:
        print("❌ FIREWORKS_API_KEY not set. Export it or set in config.")
        return False
    if not config.remote.api_key.startswith("fw_"):
        print(f"⚠️  FIREWORKS_API_KEY doesn't start with 'fw_' (got '{config.remote.api_key[:5]}...')")
    base = config.remote.base_url
    url = f"{base.rstrip('/')}/models"
    try:
        headers = {"Authorization": f"Bearer {config.remote.api_key}"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code < 500:
            models = resp.json().get("data", [])
            model_ids = {m.get("id", "") for m in models}
            selected = config.remote.model
            if selected in model_ids:
                print(f"✅ Fireworks API reachable — model '{selected}' confirmed")
            else:
                print(f"⚠️  Fireworks reachable, but model '{selected}' not listed ({len(model_ids)} available)")
            return True
        print(f"❌ Fireworks returned status {resp.status_code}: {resp.text[:200]}")
        return False
    except requests.ConnectionError:
        print("❌ Could not reach Fireworks API.")
        return False
    except Exception as exc:
        print(f"❌ Fireworks check error: {exc}")
        return False


def check_spacy() -> bool:
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("Validation test.")
        print(f"✅ spaCy model 'en_core_web_sm' loaded ({len(doc)} tokens)")
        return True
    except ImportError:
        print("❌ spaCy not installed. Run: pip install spacy")
        return False
    except OSError:
        print("❌ spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
        return False
    except Exception as exc:
        print(f"❌ spaCy error: {exc}")
        return False


def check_cache_deps() -> bool:
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        print("✅ FAISS + sentence-transformers available")
        return True
    except ImportError:
        print("⚠️  FAISS or sentence-transformers not installed. Cache Tier 0 disabled.")
        return False


def check_sklearn() -> bool:
    try:
        from sklearn.linear_model import LogisticRegression
        print("✅ scikit-learn available")
        return True
    except ImportError:
        print("⚠️  scikit-learn not installed. ANGKOR sklearn router disabled.")
        return False


def validate_all() -> bool:
    print("=" * 50)
    print("  ANGKOR + PHANTOM — Pre-flight Validation")
    print("=" * 50)
    results = [
        ("Ollama", check_ollama()),
        ("Fireworks", check_fireworks()),
        ("spaCy", check_spacy()),
        ("Cache deps", check_cache_deps()),
        ("scikit-learn", check_sklearn()),
    ]
    print("=" * 50)
    ok_count = sum(1 for _, ok in results if ok)
    print(f"  {ok_count}/{len(results)} checks passed")
    print("=" * 50)
    return all(ok for _, ok in results)


if __name__ == "__main__":
    sys.exit(0 if validate_all() else 1)
