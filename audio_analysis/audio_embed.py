#!/usr/bin/env python3
"""
audio_embed.py — analyse a folder of audio files with EmbeddingGemma 2.

EmbeddingGemma 2 (google/embeddinggemma-2) embeds text and audio into the same
768-d space, so you can:

  index       embed every audio file (chunked) into a local index
  search      find clips matching a natural-language query ("dog barking", "someone laughing")
  similar     find clips that sound like a given audio file
  classify    zero-shot label every clip against your own label list
  cluster     group clips by acoustic/semantic similarity
  duplicates  find near-duplicate recordings

Audio is decoded with ffmpeg to 16 kHz mono (what the model expects), split into
windows (default 60 s, max ~300 s per window), and each window is embedded.
A file-level embedding is the normalised mean of its window embeddings.

Usage examples:
  python audio_embed.py index  ~/recordings  --index ./idx
  python audio_embed.py search "two people arguing"  --index ./idx -k 10
  python audio_embed.py similar  ~/recordings/clip.m4a  --index ./idx
  python audio_embed.py classify --labels "speech,music,silence,traffic,birdsong"  --index ./idx
  python audio_embed.py cluster  -n 8  --index ./idx
  python audio_embed.py duplicates --threshold 0.95  --index ./idx

Set AUDIO_EMBED_FAKE=1 to run the whole pipeline with a deterministic fake
encoder (no model download) — useful for testing the plumbing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

MODEL_ID = "google/embeddinggemma-2"
SAMPLE_RATE = 16_000
MAX_WINDOW_SEC = 300          # model budget: 25 tok/s, 8192 tokens => ~327 s. Keep margin.
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus",
              ".wma", ".aiff", ".aif", ".amr", ".3gp", ".mp4", ".mkv", ".webm", ".mov"}

# --------------------------------------------------------------------------- audio I/O


def ffprobe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


def decode_audio(path: Path, start: float | None = None, dur: float | None = None) -> np.ndarray:
    """Decode any ffmpeg-readable file to float32 mono 16 kHz."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def iter_windows(wave: np.ndarray, window_sec: float, hop_sec: float):
    n = len(wave)
    win = int(window_sec * SAMPLE_RATE)
    hop = int(hop_sec * SAMPLE_RATE)
    if n <= win:
        yield 0.0, n / SAMPLE_RATE, wave
        return
    pos = 0
    while pos < n:
        seg = wave[pos:pos + win]
        if len(seg) < SAMPLE_RATE * 1.0 and pos > 0:   # drop sub-1s tail
            break
        yield pos / SAMPLE_RATE, (pos + len(seg)) / SAMPLE_RATE, seg
        pos += hop


def find_audio_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS and p.is_file())


# --------------------------------------------------------------------------- model


class Embedder:
    """Thin wrapper around SentenceTransformer (text + audio encoders only)."""

    def __init__(self, model_id: str = MODEL_ID, dim: int | None = None, device: str | None = None):
        self.dim = dim
        if os.environ.get("AUDIO_EMBED_FAKE"):
            self.model = None
            self._dim = dim or 768
            print("[fake encoder] AUDIO_EMBED_FAKE is set — embeddings are deterministic noise.", file=sys.stderr)
            return
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # bf16 on GPUs that support it; float32 elsewhere. Never float16 (NaNs).
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        t0 = time.time()
        self.model = SentenceTransformer(
            model_id,
            device=device,
            config_kwargs={"vision_config": None},        # text + audio = 570M params
            model_kwargs={"torch_dtype": dtype},
            truncate_dim=dim,
        )
        self._dim = dim or self.model.get_sentence_embedding_dimension() or 768
        print(f"[model] {model_id} loaded on {device} ({dtype}) in {time.time()-t0:.1f}s, dim={self._dim}",
              file=sys.stderr)

    # -- fake backend -----------------------------------------------------
    def _fake(self, key: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha1(key).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(self._dim).astype(np.float32)
        return v / np.linalg.norm(v)

    # -- public -----------------------------------------------------------
    def embed_audio(self, waves: list[np.ndarray], batch_size: int = 4) -> np.ndarray:
        if self.model is None:
            return np.stack([self._fake(w[:16000].tobytes()) for w in waves])
        inputs = [{"audio": {"array": w, "sampling_rate": SAMPLE_RATE}} for w in waves]
        return self.model.encode(inputs, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

    def embed_text(self, texts: list[str], prompt_name: str = "SearchQuery") -> np.ndarray:
        if self.model is None:
            return np.stack([self._fake(t.encode()) for t in texts])
        return self.model.encode(texts, prompt_name=prompt_name, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)


# --------------------------------------------------------------------------- index


class Index:
    """Chunk-level embeddings + metadata, stored as index.npz + meta.json."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.npz = folder / "index.npz"
        self.meta_path = folder / "meta.json"
        self.emb = np.zeros((0, 0), dtype=np.float32)
        self.chunks: list[dict] = []
        self.files: dict[str, dict] = {}
        self.info: dict = {}
        if self.npz.exists() and self.meta_path.exists():
            self.emb = np.load(self.npz)["emb"]
            m = json.loads(self.meta_path.read_text())
            self.chunks, self.files, self.info = m["chunks"], m["files"], m.get("info", {})

    def save(self):
        np.savez_compressed(self.npz, emb=self.emb)
        self.meta_path.write_text(json.dumps(
            {"chunks": self.chunks, "files": self.files, "info": self.info}, indent=1))

    # file-level embeddings (normalised mean of chunks)
    def file_matrix(self) -> tuple[list[str], np.ndarray]:
        names, rows = [], []
        for name, f in self.files.items():
            idx = f["chunk_ids"]
            if not idx:
                continue
            v = self.emb[idx].mean(axis=0)
            rows.append(v / (np.linalg.norm(v) + 1e-9))
            names.append(name)
        return names, (np.stack(rows) if rows else np.zeros((0, self.emb.shape[1] if self.emb.size else 0)))

    def add_file(self, name: str, chunk_meta: list[dict], chunk_emb: np.ndarray, finfo: dict):
        start = len(self.chunks)
        ids = list(range(start, start + len(chunk_meta)))
        for cm, cid in zip(chunk_meta, ids):
            cm["id"] = cid
            cm["file"] = name
        self.chunks.extend(chunk_meta)
        self.emb = chunk_emb if self.emb.size == 0 else np.vstack([self.emb, chunk_emb])
        finfo["chunk_ids"] = ids
        self.files[name] = finfo


def fmt_ts(sec: float) -> str:
    sec = int(sec)
    return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}" if sec >= 3600 else f"{sec//60:02d}:{sec%60:02d}"


def file_sig(p: Path) -> str:
    st = p.stat()
    return f"{st.st_size}:{int(st.st_mtime)}"


# --------------------------------------------------------------------------- commands


def cmd_index(a):
    root = Path(a.path).expanduser().resolve()
    files = find_audio_files(root)
    if not files:
        sys.exit(f"No audio files under {root}")
    idx = Index(Path(a.index))
    if idx.info and idx.info.get("dim") and a.dim and idx.info["dim"] != a.dim:
        sys.exit(f"Index was built with dim={idx.info['dim']}; pass --dim {idx.info['dim']} or use a new --index")
    emb = Embedder(a.model, dim=a.dim or idx.info.get("dim"), device=a.device)
    idx.info.update({"model": a.model, "dim": emb._dim, "window_sec": a.window, "hop_sec": a.hop,
                     "sample_rate": SAMPLE_RATE, "root": str(root)})

    todo = []
    for p in files:
        name = str(p.relative_to(root) if root.is_dir() else p.name)
        if name in idx.files and idx.files[name].get("sig") == file_sig(p) and not a.force:
            continue
        todo.append((name, p))
    print(f"{len(files)} audio files found, {len(todo)} to embed", file=sys.stderr)

    t0 = time.time()
    for n, (name, p) in enumerate(todo, 1):
        try:
            wave = decode_audio(p)
        except subprocess.CalledProcessError as e:
            print(f"  ! skip {name}: ffmpeg failed: {e.stderr.decode(errors='ignore')[:200]}", file=sys.stderr)
            continue
        if len(wave) < SAMPLE_RATE * 0.25:
            print(f"  ! skip {name}: shorter than 0.25 s", file=sys.stderr)
            continue
        metas, segs = [], []
        for s, e, seg in iter_windows(wave, a.window, a.hop):
            metas.append({"start": round(s, 2), "end": round(e, 2)})
            segs.append(seg)
        vecs = emb.embed_audio(segs, batch_size=a.batch_size)
        if name in idx.files:          # re-index: drop old chunks lazily by rebuilding
            _remove_file(idx, name)
        idx.add_file(name, metas, vecs.astype(np.float32),
                     {"path": str(p), "sig": file_sig(p), "duration": round(len(wave) / SAMPLE_RATE, 2)})
        print(f"  [{n}/{len(todo)}] {name}  {fmt_ts(len(wave)/SAMPLE_RATE)}  {len(segs)} window(s)", file=sys.stderr)
        if n % 10 == 0:
            idx.save()
    idx.save()
    print(f"done: {len(idx.files)} files, {len(idx.chunks)} chunks, {time.time()-t0:.0f}s -> {idx.folder}", file=sys.stderr)


def _remove_file(idx: Index, name: str):
    drop = set(idx.files[name]["chunk_ids"])
    keep = [i for i in range(len(idx.chunks)) if i not in drop]
    idx.emb = idx.emb[keep]
    idx.chunks = [idx.chunks[i] for i in keep]
    remap = {old: new for new, old in enumerate(keep)}
    for c, new in zip(idx.chunks, range(len(idx.chunks))):
        c["id"] = new
    del idx.files[name]
    for f in idx.files.values():
        f["chunk_ids"] = [remap[i] for i in f["chunk_ids"]]


def _load(a) -> tuple[Index, Embedder | None]:
    idx = Index(Path(a.index))
    if not idx.chunks:
        sys.exit(f"Index at {a.index} is empty — run `index` first")
    return idx, None


def _rank_chunks(idx: Index, q: np.ndarray, k: int, per_file: bool, out_json: bool, label: str):
    scores = idx.emb @ q
    if per_file:
        best: dict[str, int] = {}
        for i in np.argsort(-scores):
            f = idx.chunks[i]["file"]
            if f not in best:
                best[f] = int(i)
            if len(best) >= k:
                break
        hits = list(best.values())
    else:
        hits = [int(i) for i in np.argsort(-scores)[:k]]
    rows = [{"score": float(scores[i]), "file": idx.chunks[i]["file"],
             "start": idx.chunks[i]["start"], "end": idx.chunks[i]["end"]} for i in hits]
    if out_json:
        print(json.dumps({"query": label, "results": rows}, indent=1))
        return
    print(f"\n{label}")
    for r in rows:
        print(f"  {r['score']:.3f}  {r['file']}  [{fmt_ts(r['start'])}-{fmt_ts(r['end'])}]")


def cmd_search(a):
    idx, _ = _load(a)
    emb = Embedder(a.model or idx.info.get("model", MODEL_ID), dim=idx.info.get("dim"), device=a.device)
    qs = emb.embed_text(a.query, prompt_name="SearchQuery")
    for q, text in zip(qs, a.query):
        _rank_chunks(idx, q, a.k, not a.chunks, a.json, f'query: "{text}"')


def cmd_similar(a):
    idx, _ = _load(a)
    emb = Embedder(a.model or idx.info.get("model", MODEL_ID), dim=idx.info.get("dim"), device=a.device)
    p = Path(a.audio).expanduser()
    wave = decode_audio(p, a.start, a.duration)
    wave = wave[: MAX_WINDOW_SEC * SAMPLE_RATE]
    q = emb.embed_audio([wave])[0]
    _rank_chunks(idx, q, a.k, not a.chunks, a.json, f"similar to: {p.name}")


def cmd_classify(a):
    idx, _ = _load(a)
    labels = [l.strip() for l in a.labels.split(",") if l.strip()]
    emb = Embedder(a.model or idx.info.get("model", MODEL_ID), dim=idx.info.get("dim"), device=a.device)
    L = emb.embed_text(labels, prompt_name="SearchQuery")          # text labels vs audio: asymmetric
    names, F = idx.file_matrix() if not a.chunks else (None, idx.emb)
    S = F @ L.T                                                   # cosine
    P = np.exp(S / a.temperature)
    P /= P.sum(axis=1, keepdims=True)
    rows = []
    for i in range(S.shape[0]):
        order = np.argsort(-S[i])
        item = {"file": names[i]} if names else {"file": idx.chunks[i]["file"],
                                                  "start": idx.chunks[i]["start"], "end": idx.chunks[i]["end"]}
        item["label"] = labels[order[0]]
        item["confidence"] = float(P[i, order[0]])
        item["scores"] = {labels[j]: round(float(S[i, j]), 4) for j in order}
        rows.append(item)
    if a.json:
        print(json.dumps(rows, indent=1)); return
    for r in rows:
        where = r["file"] if "start" not in r else f"{r['file']} [{fmt_ts(r['start'])}-{fmt_ts(r['end'])}]"
        top = ", ".join(f"{k}={v:.3f}" for k, v in list(r["scores"].items())[:3])
        print(f"  {r['label']:<20} p={r['confidence']:.2f}  {where}    ({top})")


def cmd_cluster(a):
    idx, _ = _load(a)
    names, F = idx.file_matrix()
    if len(names) < 2:
        sys.exit("need at least 2 files to cluster")
    try:
        from sklearn.cluster import AgglomerativeClustering, KMeans
    except ImportError:
        sys.exit("pip install scikit-learn")
    if a.n:
        labels = KMeans(n_clusters=min(a.n, len(names)), n_init=10, random_state=0).fit_predict(F)
    else:   # automatic: cosine-distance threshold
        labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                         distance_threshold=1 - a.threshold).fit_predict(F)
    groups: dict[int, list[str]] = {}
    for n_, l in zip(names, labels):
        groups.setdefault(int(l), []).append(n_)
    # representative = closest to centroid; also label each cluster with user-supplied vocab if given
    emb = None
    vocab = [v.strip() for v in a.describe.split(",")] if a.describe else []
    if vocab:
        emb = Embedder(a.model or idx.info.get("model", MODEL_ID), dim=idx.info.get("dim"), device=a.device)
        V = emb.embed_text(vocab, prompt_name="SearchQuery")
    out = []
    for cid, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        M = F[[names.index(m) for m in members]]
        c = M.mean(axis=0); c /= np.linalg.norm(c) + 1e-9
        rep = members[int(np.argmax(M @ c))]
        desc = vocab[int(np.argmax(V @ c))] if vocab else None
        out.append({"cluster": cid, "size": len(members), "representative": rep, "description": desc, "files": members})
    if a.json:
        print(json.dumps(out, indent=1)); return
    for g in out:
        head = f"cluster {g['cluster']}  ({g['size']} files)  rep: {g['representative']}"
        if g["description"]:
            head += f"  ~ {g['description']}"
        print("\n" + head)
        for m in g["files"]:
            print(f"    {m}")


def cmd_duplicates(a):
    idx, _ = _load(a)
    names, F = idx.file_matrix()
    S = F @ F.T
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if S[i, j] >= a.threshold:
                pairs.append({"a": names[i], "b": names[j], "score": float(S[i, j])})
    pairs.sort(key=lambda r: -r["score"])
    if a.json:
        print(json.dumps(pairs, indent=1)); return
    if not pairs:
        print(f"no pairs above {a.threshold}"); return
    for p in pairs:
        print(f"  {p['score']:.4f}  {p['a']}  <->  {p['b']}")


def cmd_info(a):
    idx = Index(Path(a.index))
    print(json.dumps({"files": len(idx.files), "chunks": len(idx.chunks),
                      "total_audio_sec": round(sum(f["duration"] for f in idx.files.values()), 1),
                      **idx.info}, indent=1))


# --------------------------------------------------------------------------- CLI


def main(argv=None):
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH (apt install ffmpeg / pkg install ffmpeg)")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default="./audio_index", help="index folder (default ./audio_index)")
    ap.add_argument("--model", default=None, help=f"HF id or local path (default {MODEL_ID})")
    ap.add_argument("--device", default=None, help="cpu / cuda (auto)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("index", help="embed all audio under PATH"); s.set_defaults(fn=cmd_index)
    s.add_argument("path")
    s.add_argument("--window", type=float, default=60.0, help="window length in seconds (<=300)")
    s.add_argument("--hop", type=float, default=None, help="hop in seconds (default = window, no overlap)")
    s.add_argument("--dim", type=int, choices=[128, 256, 512, 768], default=None, help="Matryoshka truncation")
    s.add_argument("--batch-size", type=int, default=4)
    s.add_argument("--force", action="store_true", help="re-embed files already in the index")

    s = sub.add_parser("search", help="text -> audio search"); s.set_defaults(fn=cmd_search)
    s.add_argument("query", nargs="+")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--chunks", action="store_true", help="rank windows instead of one hit per file")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("similar", help="audio -> audio search"); s.set_defaults(fn=cmd_similar)
    s.add_argument("audio")
    s.add_argument("--start", type=float, default=None); s.add_argument("--duration", type=float, default=None)
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--chunks", action="store_true"); s.add_argument("--json", action="store_true")

    s = sub.add_parser("classify", help="zero-shot label each file"); s.set_defaults(fn=cmd_classify)
    s.add_argument("--labels", required=True, help='comma list, e.g. "speech,music,dog barking,silence"')
    s.add_argument("--temperature", type=float, default=0.05)
    s.add_argument("--chunks", action="store_true", help="label each window instead of each file")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("cluster", help="group similar files"); s.set_defaults(fn=cmd_cluster)
    s.add_argument("-n", type=int, default=None, help="number of clusters (KMeans); omit for automatic")
    s.add_argument("--threshold", type=float, default=0.80, help="cosine threshold for automatic clustering")
    s.add_argument("--describe", default=None, help="comma list of words to name clusters with")
    s.add_argument("--json", action="store_true")

    s = sub.add_parser("duplicates", help="near-duplicate files"); s.set_defaults(fn=cmd_duplicates)
    s.add_argument("--threshold", type=float, default=0.95); s.add_argument("--json", action="store_true")

    s = sub.add_parser("info", help="index stats"); s.set_defaults(fn=cmd_info)

    a = ap.parse_args(argv)
    if a.cmd == "index":
        a.window = min(a.window, MAX_WINDOW_SEC)
        a.hop = a.hop or a.window
    a.fn(a)


if __name__ == "__main__":
    main()
