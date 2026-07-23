"""
Entity name normalization to canonical IDs.
Priority: static alias table → known compound map → slugify fallback.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from config import CANONICAL_IDS_PATH

# ── Gene / protein alias table ────────────────────────────────────────────────
_GENE_ALIASES: dict[str, str] = {
    # TDP-43 / TARDBP
    "TDP-43": "TARDBP", "TDP43": "TARDBP", "tdp-43": "TARDBP", "tdp43": "TARDBP",
    "TAR DNA-binding protein 43": "TARDBP",
    "TAR DNA binding protein 43": "TARDBP",
    "TAR DNA-binding protein": "TARDBP",
    # FUS
    "FUS/TLS": "FUS", "TLS": "FUS", "TLS/FUS": "FUS", "fus": "FUS",
    "fused in sarcoma": "FUS",
    # C9orf72
    "C9ORF72": "C9orf72", "c9orf72": "C9orf72", "C9": "C9orf72",
    "chromosome 9 open reading frame 72": "C9orf72",
    # SOD1
    "SOD-1": "SOD1", "sod1": "SOD1",
    "superoxide dismutase 1": "SOD1",
    "Cu/Zn-superoxide dismutase": "SOD1",
    "copper-zinc superoxide dismutase": "SOD1",
    # ATXN2
    "ataxin-2": "ATXN2", "ataxin 2": "ATXN2", "SCA2": "ATXN2",
    # Optineurin
    "optineurin": "OPTN",
    # Ubiquilin
    "ubiquilin-2": "UBQLN2", "ubiquilin2": "UBQLN2", "ubiquilin 2": "UBQLN2",
    # SQSTM1 / p62
    "p62": "SQSTM1", "sequestosome-1": "SQSTM1", "sequestosome 1": "SQSTM1",
    # Profilin
    "profilin 1": "PFN1", "profilin-1": "PFN1",
    # Dynactin
    "dynactin": "DCTN1", "dynactin 1": "DCTN1",
    # Matrin
    "matrin 3": "MATR3", "matrin-3": "MATR3",
    # Other
    "angiogenin": "ANG",
    "senataxin": "SETX",
}

# ── Compound alias table ───────────────────────────────────────────────────────
_COMPOUND_ALIASES: dict[str, str] = {
    "riluzole": "riluzole", "Riluzole": "riluzole",
    "edaravone": "edaravone", "Edaravone": "edaravone", "MCI-186": "edaravone",
    "tofersen": "tofersen", "Tofersen": "tofersen",
    "BIIB067": "tofersen", "biib067": "tofersen",
    "AMX0035": "AMX0035", "amx0035": "AMX0035",
    "sodium phenylbutyrate": "AMX0035",
    "tauroursodeoxycholic acid": "AMX0035",
    "TUDCA": "AMX0035",
    "masitinib": "masitinib", "Masitinib": "masitinib", "AB1010": "masitinib",
    "bosutinib": "bosutinib", "Bosutinib": "bosutinib", "SKI-606": "bosutinib",
    "mexiletine": "mexiletine", "Mexiletine": "mexiletine",
    "memantine": "memantine", "Memantine": "memantine",
    "rasagiline": "rasagiline", "Rasagiline": "rasagiline",
    "NurOwn": "NurOwn", "MSC-NTF": "NurOwn",
    "ozanezumab": "ozanezumab",
}

# ── Mechanism normalization ────────────────────────────────────────────────────
_MECHANISM_ALIASES: dict[str, str] = {
    "glutamate excitotoxicity": "glutamate_excitotoxicity",
    "excitotoxicity": "glutamate_excitotoxicity",
    "glutamatergic excitotoxicity": "glutamate_excitotoxicity",
    "oxidative stress": "oxidative_stress",
    "reactive oxygen species": "oxidative_stress",
    "ROS": "oxidative_stress",
    "neuroinflammation": "neuroinflammation",
    "microglial activation": "neuroinflammation",
    "astrocyte activation": "neuroinflammation",
    "protein aggregation": "protein_aggregation",
    "protein misfolding": "protein_aggregation",
    "protein inclusions": "protein_aggregation",
    "RNA metabolism": "RNA_metabolism_dysfunction",
    "RNA processing": "RNA_metabolism_dysfunction",
    "RNA-binding protein dysfunction": "RNA_metabolism_dysfunction",
    "stress granules": "RNA_metabolism_dysfunction",
    "mitochondrial dysfunction": "mitochondrial_dysfunction",
    "mitochondrial impairment": "mitochondrial_dysfunction",
    "axonal transport": "axonal_transport_defect",
    "axonal transport defect": "axonal_transport_defect",
    "autophagy": "autophagy_impairment",
    "mitophagy": "autophagy_impairment",
    "ubiquitin proteasome": "autophagy_impairment",
    "TDP-43 pathology": "TDP43_pathology",
    "TDP-43 aggregation": "TDP43_pathology",
    "TDP-43 mislocalization": "TDP43_pathology",
    "antisense oligonucleotide": "antisense_oligonucleotide",
    "ASO": "antisense_oligonucleotide",
    "gene therapy": "gene_therapy",
    "stem cell": "stem_cell_therapy",
    "neurodegeneration": "neurodegeneration",
    "apoptosis": "apoptosis",
    "DNA damage": "DNA_damage_repair",
}


def guess_entity_type(name: str) -> str:
    """Best-effort entity type inference from name. Used when type metadata is unavailable."""
    n = name.strip()
    if n.upper() in _GENE_ALIASES or n in _GENE_ALIASES:
        return "Gene"
    if n in _COMPOUND_ALIASES:
        return "Compound"
    return "Protein"


def normalize_entity(name: str, entity_type: str) -> str:
    """Return a canonical_id string in the form '<prefix>:<canonical_name>'."""
    canonical = _resolve_name(name.strip(), entity_type)
    return f"{_prefix(entity_type)}:{canonical}"


def _resolve_name(name: str, entity_type: str) -> str:
    t = entity_type.lower()

    if t == "gene":
        return _GENE_ALIASES.get(name) or _GENE_ALIASES.get(name.upper()) or name.upper()

    if t == "protein":
        gene_hit = _GENE_ALIASES.get(name) or _GENE_ALIASES.get(name.upper())
        if gene_hit:
            return gene_hit
        return name[0].upper() + name[1:] if name else name

    if t == "compound":
        return _COMPOUND_ALIASES.get(name) or _slugify(name)

    if t == "mechanism":
        return _MECHANISM_ALIASES.get(name) or _slugify(name)

    return _slugify(name)


def _prefix(entity_type: str) -> str:
    return {
        "gene": "gene",
        "protein": "protein",
        "compound": "compound",
        "pathway": "pathway",
        "phenotype": "phenotype",
        "mechanism": "mechanism",
    }.get(entity_type.lower(), "entity")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


class CanonicalRegistry:
    """Persistent name→canonical_id mapping written to canonical_ids.json."""

    def __init__(self, path: Path = CANONICAL_IDS_PATH) -> None:
        self.path = path
        self._data: dict[str, str] = {}
        if path.exists():
            self._data = json.loads(path.read_text())

    def resolve(self, name: str, entity_type: str) -> str:
        key = f"{entity_type.lower()}:{name}"
        if key not in self._data:
            self._data[key] = normalize_entity(name, entity_type)
        return self._data[key]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

    def __len__(self) -> int:
        return len(self._data)
