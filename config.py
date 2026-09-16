from pathlib import Path

# Models
SYNTHESIS_MODEL = "claude-sonnet-4-6"
EXTRACTION_MODEL = "claude-haiku-4-5-20251001"
# Tier-2 eval judge — deliberately a different, stronger model than SYNTHESIS_MODEL
# so we don't grade the answer with the model that wrote it (self-preference bias).
JUDGE_MODEL = "claude-opus-4-8"

# External API endpoints
CTGOV_BASE = "https://clinicaltrials.gov/api/v2/studies"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
HGNC_REST_BASE = "https://rest.genenames.org"
PUBCHEM_REST_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Data paths
DATA_DIR = Path(__file__).parent / "data"
PAPERS_PATH = DATA_DIR / "papers" / "papers.jsonl"
TRIALS_PATH = DATA_DIR / "trials" / "trials.jsonl"
ENTITIES_PATH = DATA_DIR / "extracted" / "entities.jsonl"
CANONICAL_IDS_PATH = DATA_DIR / "extracted" / "canonical_ids.json"
EXTRACTION_PROGRESS_PATH = DATA_DIR / "extracted" / ".progress.json"
EXTRACTION_BATCH_STATE_PATH = DATA_DIR / "extracted" / ".batch_state.json"
GRAPH_PICKLE_PATH = DATA_DIR / "graph" / "als_graph.pkl"
GRAPH_JSON_PATH = DATA_DIR / "graph" / "als_graph.json"
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_COLLECTION = "als_papers"

# Runtime data lives OUTSIDE git (see .gitignore) and is served from a HF dataset repo:
# scripts/upload_data.py pushes it there, app._ensure_data() pulls it on Space startup.
# Both sides read this one constant so the upload/download targets can never drift.
HF_DATASET_REPO = "KevinIsCoding/candle-fire-data"

# Experimental therapy landscape (offline-built, committed to git — small)
LANDSCAPE_PATH = DATA_DIR / "landscape" / "landscape.json"
LANDSCAPE_PROGRESS_PATH = DATA_DIR / "landscape" / ".progress.json"
LANDSCAPE_BATCH_STATE_PATH = DATA_DIR / "landscape" / ".batch_state.json"
THERAPY_CLASSES_PATH = DATA_DIR / "seeds" / "therapy_classes.json"
THERAPY_GOLD_PATH = DATA_DIR / "seeds" / "therapy_gold.json"
# Mechanism classification (v2): frontier model, grounded + multi-label + abstaining.
LANDSCAPE_MODEL = "claude-opus-4-8"
LANDSCAPE_MIN_CONFIDENCE = 0.55   # τ — drop LLM mechanisms below this confidence
LANDSCAPE_XCHECK_MIN_COSINE = 0.20  # BioLORD guard: drop a mechanism whose justification↔class cosine is below this (loose; catches gross mismatch)
LANDSCAPE_EVIDENCE_ABSTRACTS = 4  # top abstracts retrieved per therapy as MoA evidence

# Seed entity files
MANUAL_SEEDS_PATH = DATA_DIR / "seeds" / "manual_seeds.json"
DERIVED_SEEDS_PATH = DATA_DIR / "seeds" / "derived_seeds.json"
SEED_PROMOTION_THRESHOLD = 5   # min papers for entity to become a derived seed
REFRESH_STATE_PATH = DATA_DIR / ".refresh_state.json"

# PubMed ingestion defaults
PUBMED_BASE_QUERY = (
    '"amyotrophic lateral sclerosis"[MeSH Major Topic] '
    "AND hasabstract[text]"
)
PUBMED_DEFAULT_QUERY = PUBMED_BASE_QUERY  # no date cap — fetch all 19k+ ALS papers
PUBMED_REFRESH_QUERY_TEMPLATE = (
    '"amyotrophic lateral sclerosis"[MeSH Major Topic] '
    'AND ("{since_date}"[PDAT]:"3000"[PDAT]) '
    "AND hasabstract[text]"
)
PUBMED_DEFAULT_MAX = 20000
PUBMED_BATCH_SIZE = 200  # PMIDs per Entrez efetch call

# Entity extraction
EXTRACTION_BATCH_SIZE = 20  # papers per Claude call
EXTRACTION_WORKERS = 8      # parallel Claude calls (Haiku limit: 1000 RPM on paid tier)

# RAG — retrieval counts per stage
CHROMA_N_RESULTS = 10          # legacy default (kept for backward compat)
CHROMA_ENTITY_N_RESULTS = 15   # legacy default (kept for backward compat)
RETRIEVAL_SEMANTIC_N = 30      # semantic search candidate pool
RETRIEVAL_ENTITY_N = 30        # entity search candidate pool
RETRIEVAL_ENTITY_QUERY_CAP = 12  # max entity names to query individually

# RRF merge
RRF_K = 10       # lower k → stronger rank differentiation (k=60 is too flat for 30-item lists)
RRF_TOP_N = 20   # candidates passed to cross-encoder

# Cross-encoder reranking
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_TOP_N = 15  # final papers sent to Claude for synthesis

# Knowledge graph
KG_EXPANSION_HOPS = 1  # hops for query entity expansion
KG_MIN_EDGE_CONFIDENCE = 0.3  # edges below this are excluded from traversal

# ALS condition synonyms for ClinicalTrials.gov queries
ALS_CONDITION_TERMS = [
    "Amyotrophic Lateral Sclerosis",
    "ALS",
    "Motor Neuron Disease",
    "Lou Gehrig's Disease",
]
