from pathlib import Path

# Models
SYNTHESIS_MODEL = "claude-sonnet-4-6"
EXTRACTION_MODEL = "claude-haiku-4-5-20251001"

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
GRAPH_PICKLE_PATH = DATA_DIR / "graph" / "als_graph.pkl"
GRAPH_JSON_PATH = DATA_DIR / "graph" / "als_graph.json"
CHROMA_DIR = DATA_DIR / "chroma"
CHROMA_COLLECTION = "als_papers"

# PubMed ingestion defaults
PUBMED_DEFAULT_QUERY = (
    '"amyotrophic lateral sclerosis"[MeSH Major Topic] '
    'AND ("2018"[PDAT]:"2024"[PDAT]) '
    "AND hasabstract[text]"
)
PUBMED_DEFAULT_MAX = 500
PUBMED_BATCH_SIZE = 200  # PMIDs per Entrez efetch call

# Entity extraction
EXTRACTION_BATCH_SIZE = 10  # papers per Claude call

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

# ALS seed entities — pre-populate the graph before paper-derived extraction
ALS_SEED_ENTITIES = {
    "genes": [
        "SOD1", "TARDBP", "FUS", "C9orf72", "ATXN2",
        "TBK1", "OPTN", "UBQLN2", "VCP", "NEK1",
        "ANG", "SETX", "SIGMAR1", "CHCHD10", "MATR3",
    ],
    "proteins": [
        "TDP-43", "FUS protein", "SOD1 protein", "Alsin",
        "Optineurin", "p62", "Ubiquilin-2",
    ],
    "compounds": [
        "riluzole", "edaravone", "tofersen", "AMX0035",
        "mexiletine", "masitinib", "bosutinib",
    ],
    "mechanisms": [
        "glutamate excitotoxicity", "oxidative stress",
        "neuroinflammation", "protein aggregation",
        "RNA metabolism dysfunction", "mitochondrial dysfunction",
        "axonal transport defect", "autophagy impairment",
    ],
    "phenotypes": [
        "upper motor neuron degeneration", "lower motor neuron degeneration",
        "bulbar onset ALS", "spinal onset ALS",
        "frontotemporal dementia", "respiratory failure",
    ],
}

# ALS condition synonyms for ClinicalTrials.gov queries
ALS_CONDITION_TERMS = [
    "Amyotrophic Lateral Sclerosis",
    "ALS",
    "Motor Neuron Disease",
    "Lou Gehrig's Disease",
]
