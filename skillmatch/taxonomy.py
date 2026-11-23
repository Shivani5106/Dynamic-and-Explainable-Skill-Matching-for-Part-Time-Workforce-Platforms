"""Skill taxonomy: 8 clusters x 6 skills, plus a skill-similarity graph.

The graph plays the role of the "semantic" layer: identical skill = 1.0,
same cluster = 0.5, a few hand-set cross-cluster links = 0.2-0.3.
(A sentence-embedding model such as SBERT can replace SIM; see embed.py.)
"""
import numpy as np

CLUSTERS = {
    "web": ["HTML", "CSS", "JavaScript", "React", "Node.js", "REST APIs"],
    "data": ["Python", "SQL", "Excel", "Pandas", "Data Visualization", "Statistics"],
    "design": ["Photoshop", "Illustrator", "Figma", "UI Design", "Branding", "Video Editing"],
    "writing": ["Copywriting", "Content Writing", "Proofreading", "SEO", "Translation", "Technical Writing"],
    "support": ["Customer Support", "Data Entry", "Email Handling", "CRM Tools", "Scheduling", "Phone Support"],
    "hospitality": ["Barista", "Food Prep", "Event Setup", "Cashiering", "Serving", "Housekeeping"],
    "logistics": ["Delivery", "Warehouse", "Inventory", "Forklift", "Packing", "Route Planning"],
    "teaching": ["Math Tutoring", "English Tutoring", "Science Tutoring", "Test Prep", "Curriculum", "Child Care"],
}

CLUSTER_NAMES = list(CLUSTERS)
SKILL_NAMES = [s for skills in CLUSTERS.values() for s in skills]
N_CLUSTERS = len(CLUSTER_NAMES)
N_SKILLS = len(SKILL_NAMES)
SKILL_IDX = {s: i for i, s in enumerate(SKILL_NAMES)}
CLUSTER_OF = np.array([ci for ci, skills in enumerate(CLUSTERS.values()) for _ in skills])
CLUSTER_SKILLS = [np.where(CLUSTER_OF == ci)[0] for ci in range(N_CLUSTERS)]

_CROSS = [
    ("Python", "JavaScript", 0.25), ("UI Design", "CSS", 0.30),
    ("Data Visualization", "JavaScript", 0.20), ("Email Handling", "Copywriting", 0.20),
    ("Inventory", "Excel", 0.25), ("Cashiering", "Data Entry", 0.20),
    ("Child Care", "Math Tutoring", 0.20), ("Technical Writing", "Statistics", 0.10),
]


def build_similarity():
    sim = np.eye(N_SKILLS)
    same = CLUSTER_OF[:, None] == CLUSTER_OF[None, :]
    sim = np.where(same & ~np.eye(N_SKILLS, dtype=bool), 0.5, sim)
    for a, b, v in _CROSS:
        sim[SKILL_IDX[a], SKILL_IDX[b]] = sim[SKILL_IDX[b], SKILL_IDX[a]] = v
    return sim


SIM = build_similarity()
