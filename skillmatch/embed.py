"""Optional: replace the skill-graph similarity with SBERT embeddings.

The sandbox used to build this prototype could not download model weights, so the reported results
use the skill-graph similarity in taxonomy.py.  On a machine with internet access:

    pip install sentence-transformers
    from skillmatch import embed, taxonomy
    taxonomy.SIM[:] = embed.sbert_similarity()      # then re-run run_experiments.py

Re-running will change the numbers; report the SBERT run if you use it.
"""
import numpy as np
from .taxonomy import SKILL_NAMES


def sbert_similarity(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name)
    e = m.encode(SKILL_NAMES, normalize_embeddings=True)
    sim = np.clip(e @ e.T, 0, 1)
    np.fill_diagonal(sim, 1.0)
    return sim
