# PRFAQ: Candle-Fire — ALS Research Intelligence for Physicians

---

## Press Release

**FOR IMMEDIATE RELEASE**

### Candle-Fire Gives ALS Physicians Instant Research Intelligence, Cutting Hours of Literature Review to Seconds

*New AI-powered tool synthesizes 500+ peer-reviewed ALS papers and live clinical trial data into a single research landscape report, grounded in a biomedical knowledge graph*

**New Hyde Park, NY** — 12/30/2026 Today, the team behind Beacon announced the launch of Candle-Fire, a research intelligence platform that helps physicians evaluating ALS (amyotrophic lateral sclerosis) clinical trials understand the full scientific context behind a trial's mechanism; to help researcher to understand the disease pathology and existing experiment in order to better locate disease target.

ALS is one of the most aggressive and least understood neurodegenerative diseases. With over 100 active clinical trials testing targets ranging from SOD1 gene silencing to neuroinflammation modulators, physicians face a mounting challenge: how do you evaluate a trial's scientific credibility when the evidence base spans thousands of papers across genetics, cell biology, and clinical pharmacology?

Today, that process takes hours. A neurologist reviewing a tofersen trial must manually search PubMed, cross-reference SOD1 biology, understand antisense oligonucleotide mechanisms, and trace the lineage of evidence from animal models to Phase 3 data — all before forming a clinical opinion. Most physicians simply do not have that time.

Candle-Fire changes this. A physician types a natural-language question — "What is the evidence for tofersen targeting SOD1 in ALS?" — and receives a structured research synthesis in seconds: key mechanisms, genes and proteins involved, evidence strength, top-cited papers with links, and related clinical trials. Every claim is grounded in peer-reviewed literature and cited with PubMed IDs.

The system combines two layers of intelligence. A biomedical knowledge graph connects genes, proteins, compounds, pathways, and clinical trials extracted from 500 curated ALS papers, enabling the system to surface related biology that a keyword search would miss. A vector retrieval layer finds the most relevant passages from those papers and weights them by citation count, surfacing high-impact evidence over peripheral findings. Claude Sonnet synthesizes both into a physician-readable report, streamed in real time.

Looking further ahead, the team envisions Candle-Fire playing a longer-term role in accelerating compound discovery. When researchers can rapidly map which biological pathways remain poorly targeted, which compound classes have already been tested and failed, and where the strongest mechanistic evidence clusters, they gain a clearer picture of the white space worth pursuing. A system that continuously synthesizes the growing ALS literature — tracking which gene-compound relationships are well-evidenced and which are hypothetical — can serve as an early signal layer for drug discovery teams deciding where to focus preclinical investment. As the knowledge graph matures and the corpus expands, Candle-Fire's relationship network between compounds, targets, and mechanisms has the potential to surface novel combination hypotheses that no single research team could derive from manual review alone.

"ALS research moves fast, but the tools physicians use to stay current haven't kept up," said the Candle-Fire team. "We built this because the gap between what the research community knows and what a treating neurologist can practically access in a clinic visit is too wide."

Candle-Fire is available as a web app on HuggingFace Spaces. It is free to use and open-source under the MIT License.

---

## Frequently Asked Questions

### Customer FAQs

**Q: Who is Candle-Fire for?**

A: Candle-Fire is built for neurologists, clinical researchers, and trial coordinators who work with ALS patients. The primary user is a physician who has encountered a clinical trial — either as a potential referral for a patient or as an investigator evaluating participation — and wants to quickly understand the research evidence behind that trial's mechanism without spending hours on PubMed.

Secondary users include ALS researchers who want a fast synthesis of what is known about a specific target, and patient advocates who want to understand the science behind trials they are tracking.

---

**Q: What kinds of questions can I ask?**

A: Candle-Fire is designed for mechanism- and target-level questions grounded in ALS biology. Examples:

- *"What is the evidence for SOD1 as a therapeutic target in ALS?"*
- *"What compounds target glutamate excitotoxicity in ALS and what is their clinical status?"*
- *"What is the role of TDP-43 aggregation in ALS pathology?"*
- *"What does the evidence say about C9orf72 repeat expansion and ALS/FTD overlap?"*
- *"What antisense oligonucleotides are in ALS trials and what are they targeting?"*
- *"What is the mechanism of AMX0035 and how strong is the evidence?"*

Questions outside the ALS research corpus (e.g., about other diseases, or about specific patients) are outside scope.

---

**Q: How current is the research it draws on?**

A: The current corpus covers approximately 500 curated ALS papers published between 2018 and 2024, prioritizing high-impact and highly cited work indexed in PubMed. Clinical trial data is fetched live from ClinicalTrials.gov at the time of the query. We plan to add automated corpus updates to keep the knowledge base current with new publications.

---

**Q: How do I know the answers are accurate?**

A: Every factual claim in Candle-Fire's output is accompanied by a PubMed citation. The system is instructed to cite PMIDs for all statements and to acknowledge uncertainty when evidence is limited or conflicting. Citation counts from Semantic Scholar are used to weight evidence — a finding supported by a paper cited 500 times is surfaced more prominently than one cited 5 times.

That said, Candle-Fire is a **research synthesis tool, not a clinical decision support system**. It is designed to give physicians a head start on literature review, not to replace it. All outputs include a disclaimer reminding users to verify findings in primary sources.

---

**Q: What is the difference between Candle-Fire and just searching PubMed or ChatGPT?**

A: Three key differences:

1. **Citations are grounded in real papers.** PubMed search returns papers but requires you to read and synthesize them yourself. Generic LLMs (like ChatGPT without retrieval) may hallucinate citations or miss recent work. Candle-Fire retrieves actual papers from its corpus and cites them by PMID.

2. **The knowledge graph surfaces related biology.** A keyword search for "tofersen" only finds papers that mention tofersen. The Candle-Fire knowledge graph knows that tofersen targets SOD1, which is connected to oxidative stress, which is connected to riluzole's mechanism — so a query about tofersen can surface the broader SOD1 and oxidative stress literature automatically.

3. **Evidence is weighted by citation impact.** Not all papers are equal. A landmark study cited 800 times and a small case series cited 3 times both appear in PubMed search results. Candle-Fire prioritizes high-citation evidence.

---

**Q: Is my query data stored or used for training?**

A: No. Queries are passed to the Anthropic Claude API for synthesis and are subject to Anthropic's data usage policies. The tool does not log or store physician queries beyond what is needed for a single session.

---

**Q: Is it free?**

A: Yes. Candle-Fire is open-source (MIT License) and the hosted version on HuggingFace Spaces is free to use.

---

### Internal FAQs

**Q: Why ALS specifically? Why not all neurodegenerative diseases?**

A: Three reasons, and none of them are about the genetics being simple — because they are not.

**An active but under-resourced research frontier.** ALS is a disease where the research community is working hard but pharmaceutical investment lags far behind. Over 100 clinical trials are recruiting at any given time, yet only two drugs (riluzole, edaravone) have reached broad approval in decades. That gap between scientific activity and clinical translation is exactly where better research synthesis tools create value — helping physicians and researchers navigate a crowded and fast-moving evidence base without deep institutional resources.

**A cruel and fast disease that demands urgency.** ALS is uniformly fatal, typically within 2–5 years of symptom onset, with no disease-modifying cure. Patients deteriorate rapidly. Physicians evaluating trial options for a patient diagnosed today do not have months to conduct literature review — they need synthesis now. The disease's brutality is the strongest argument for tools that compress hours of work into seconds.

**Scientific complexity that rewards structured intelligence.** Contrary to a common misconception, 90% of ALS cases are sporadic — no identified genetic cause. Only ~10% are familial, and within that group, SOD1, TARDBP, FUS, and C9orf72 are the primary genes. The vast majority of patients have ALS arising from an incompletely understood convergence of genetic susceptibility, environmental exposure, and cellular stress mechanisms. This is precisely why a knowledge graph that maps pathological mechanisms, not just genes, is valuable. The biology that connects protein aggregation, neuroinflammation, oxidative stress, and RNA dysregulation is the real frontier — and that frontier is too complex to navigate by keyword search alone.

The architecture generalizes: replacing the ALS paper corpus with a Parkinson's or Alzheimer's corpus requires only re-running the ingestion and extraction pipeline with a different PubMed query. ALS is the pilot, not the ceiling.

---

**Q: How does the knowledge graph get built? Can it be trusted?**

A: The KG is built in two phases. First, a curated seed of known ALS entities (15 genes, 7 proteins, 7 compounds, 8 mechanisms, 6 phenotypes) is loaded from `config.py` — these are manually validated facts. Second, Claude Sonnet extracts additional entities and relationships from each paper abstract, which are then normalized to canonical IDs (HGNC gene symbols, PubChem compound IDs) before being merged into the graph.

The extraction is not perfect — LLM-based NER at abstract level achieves roughly 80-85% precision on named biomedical entities. We mitigate this through: (1) confidence thresholding on extracted edges, (2) requiring multiple paper corroboration for high-confidence edges, and (3) citation-count weighting so edges supported by high-impact papers carry more weight. A physician-reviewed QA pass on the seed entities provides a correctness baseline.

---

**Q: Why ChromaDB and NetworkX instead of a managed vector DB and Neo4j?**

A: For the current scale (500 papers, ~2,000 graph nodes), the lightweight stack is the right call. ChromaDB is SQLite-backed and requires no separate service — it runs in-process and is trivially deployable on HuggingFace Spaces. NetworkX loads the full graph from a pickle file in milliseconds at startup and keeps it in memory across requests.

The architecture is designed so that migrating to Pinecone + Neo4j is a two-file change (`rag/retriever.py` and `graph/serializer.py`) if we scale to 20,000+ papers. We are not solving a scale problem that does not yet exist.

---

**Q: What is the relationship between Candle-Fire and Beacon?**

A: Beacon helps ALS patients find clinical trials near them. Candle-Fire helps physicians understand the science behind those trials. They are complementary: a patient might find a trial on Beacon and bring it to their neurologist, who can then use Candle-Fire to evaluate it.

Both projects share the same architectural conventions (Anthropic Claude, Gradio, HuggingFace Spaces, Python + uv), the same LLM provider abstraction, and the same ClinicalTrials.gov data source. Future work may allow the two systems to share trial data directly.

---

**Q: What are the biggest risks?**

A: Three risks worth watching:

1. **LLM hallucination in synthesis.** Even with retrieved context, Claude can occasionally mischaracterize a paper's findings. Mitigation: strict citation requirements in the synthesis prompt, and a UI that displays source paper titles alongside every claim so physicians can spot-check.

2. **Knowledge graph staleness.** ALS research moves fast. A relationship that was accurate in 2022 may be obsolete by 2025. Mitigation: plan automated re-ingestion and re-extraction on a quarterly cadence; surface publication dates prominently.

3. **Physician trust calibration.** Physicians may over-trust or under-trust the system. Under-trust means they ignore it; over-trust means they act on hallucinated citations. Mitigation: prominent disclaimers, PMIDs that link directly to PubMed, and explicit uncertainty language in the synthesis prompt ("evidence is limited," "results have not been replicated," etc.).

---

**Q: What does the roadmap look like after launch?**

A: In priority order:

1. **Expand corpus to 2,000+ papers** — extend PubMed query window and add landmark pre-2018 ALS papers by curated PMID list.
2. **Add trial-to-research linking** — when a physician queries a specific NCT ID, surface all papers directly relevant to that trial's mechanism.
3. **Protein structure and interaction data** — integrate AlphaFold binding data and STRING protein interaction network to enrich the KG's protein-level edges.
4. **Generalize to other motor neuron diseases** — SMA and Huntington's as the next corpus candidates (both already supported in Beacon's disease registry).
5. **Physician feedback loop** — thumbs up/down on individual citations to improve retrieval ranking over time.
