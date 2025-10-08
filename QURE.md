# **QURE: Multi-Agent Intelligence System \- Claude Code Design Document**

## **Executive Summary**

**QURE (Quality & Uncertainty Resolution Engine)** is a production-grade, multi-agent AI system that automates complex back-office exception resolution across Insurance, Healthcare, Finance, and Manufacturing verticals. This design document specifies the architecture, agent responsibilities, and implementation plan for a compelling demo that showcases AI-driven automation where traditional RPA fails.

**Core Thesis**: Treat resolution as a **policy problem** over a **knowledge substrate** (vectors \+ graphs \+ features) with **assurance/uncertainty** at every step. LLMs handle semantics, rules provide auditability, algorithms deliver precision, ML generalizes, and a learning policy arbitrates.

---

## **1\. System Architecture**

### **1.1 High-Level Design**

┌─────────────────────────────────────────────────────────────┐  
│                     QURE Control Plane                       │  
│                    (Orchestration Agent)                     │  
└─────────────────┬───────────────────────────────────────────┘  
                  │  
                  ├─────► Retriever Agent (data ingestion)  
                  │  
                  ├─────► Data Agent (normalization, UDI)  
                  │  
                  ├─────► Knowledge Substrate  
                  │       ├─ Vector Store (embeddings)  
                  │       ├─ Graph DB (entities/relationships)  
                  │       └─ Feature Store (structured features)  
                  │  
                  ├─────► Reasoning Mesh  
                  │       ├─ Rules Engine Agent  
                  │       ├─ Algorithm Agent  
                  │       ├─ ML Model Agent  
                  │       └─ GenAI Reasoner Agent  
                  │  
                  ├─────► Assurance Agent (uncertainty scoring)  
                  │  
                  ├─────► Policy Agent (decision fusion)  
                  │  
                  ├─────► Action Agent (execute/enrich/route)  
                  │  
                  └─────► Learning Agent (feedback loop)

### **1.2 Agent Communication Protocol**

All agents communicate via:

* **Async message queues** (Redis/RabbitMQ for local dev, Kafka for production)  
* **gRPC** for synchronous RPC when needed  
* **Shared state** via the Knowledge Substrate  
* **JSON schemas** for all inter-agent messages

---

## **2\. Agent Specifications**

### **2.1 Orchestration Agent (Control Plane)**

**Responsibility**: Coordinate agent execution, manage workflows, enforce governance constraints

**Key Operations**:

* Route incoming cases to appropriate agent chains  
* Manage agent execution DAG (which agents run when)  
* Apply hard constraints (mandatory rules, compliance gates)  
* Track execution state and handle failures  
* Emit audit logs for every decision

**Tech Stack**:

* Python 3.11+  
* Temporal.io or Apache Airflow for workflow orchestration  
* Redis for state management

**Key Files**:

agents/orchestration/  
├── orchestrator.py          \# Main control logic  
├── workflow\_definitions.py  \# DAG specifications per use case  
├── constraint\_engine.py     \# Governance rules  
└── audit\_logger.py         \# Immutable decision log

---

### **2.2 Retriever Agent**

**Responsibility**: Fetch data from heterogeneous sources (ERP, CRM, emails, PDFs, APIs)

**Key Operations**:

* Connect to data sources via adapters (SAP, Salesforce, S3, Gmail, etc.)  
* Handle auth, retries, rate limiting  
* Stream large documents  
* Emit raw data to Data Agent

**Tech Stack**:

* Python with async I/O (aiohttp, asyncio)  
* Source-specific SDKs (boto3 for S3, google-api-python-client, etc.)  
* Document parsers (PyPDF2, python-docx, BeautifulSoup)

**Key Files**:

agents/retriever/  
├── retriever.py            \# Main retriever logic  
├── connectors/  
│   ├── s3\_connector.py  
│   ├── gmail\_connector.py  
│   ├── erp\_connector.py    \# SAP, Oracle, etc.  
│   └── crm\_connector.py    \# Salesforce, Dynamics  
├── parsers/  
│   ├── pdf\_parser.py  
│   ├── email\_parser.py  
│   └── ocr\_handler.py      \# Tesseract/Textract for scanned docs  
└── cache\_layer.py          \# Redis cache for deduplication

---

### **2.3 Data Agent (Universal Data Integrator)**

**Responsibility**: Parse, normalize, link, and enrich raw data into the Knowledge Substrate

**Key Operations**:

* Extract entities (NER: parties, dates, amounts, codes)  
* Normalize formats (dates, currency, addresses)  
* Resolve references (link claim ID to party, payment to claim)  
* Compute embeddings for text chunks  
* Build property graph edges

**Tech Stack**:

* spaCy or Hugging Face transformers for NER  
* OpenAI/Cohere embeddings API (or local sentence-transformers)  
* Graph construction library (NetworkX for dev, Neo4j driver for prod)  
* Pandas for structured data transforms

**Key Files**:

agents/data/  
├── data\_agent.py              \# Main UDI pipeline  
├── entity\_extraction.py       \# NER models  
├── normalization.py           \# Date/currency/address standardization  
├── embedding\_engine.py        \# Text → vectors  
├── graph\_builder.py           \# Construct nodes/edges  
└── feature\_engineering.py     \# Compute point-in-time features

---

### **2.4 Knowledge Substrate (Shared State Layer)**

**Responsibility**: Unified storage for vectors, graphs, and features

**Components**:

1. **Vector Store**: Semantic search over document chunks

   * Options: ChromaDB (local), Pinecone/Weaviate (cloud)  
2. **Graph Database**: Entity relationships and lineage

   * Options: Neo4j (production), NetworkX (dev)  
3. **Feature Store**: Versioned, point-in-time features

   * Options: Feast, Tecton, or custom Postgres \+ DuckDB  
4. **Evidence Links**: Every claim has pointers to source text spans

**Tech Stack**:

* ChromaDB or Weaviate for vectors  
* Neo4j for graphs  
* Postgres \+ JSON columns for features  
* Redis for hot cache

**Key Files**:

substrate/  
├── vector\_store.py       \# Embedding CRUD operations  
├── graph\_store.py        \# Neo4j client wrapper  
├── feature\_store.py      \# Feature retrieval/versioning  
└── evidence\_tracker.py   \# Source span linking

---

### **2.5 Rules Engine Agent**

**Responsibility**: Apply deterministic business logic, compliance checks, validation rules

**Key Operations**:

* Evaluate rules DSL (custom or use Open Policy Agent/Rego)  
* Return `{pass, fail, needs_evidence}` with rationale  
* Enforce hard constraints (SOX controls, HIPAA, state insurance regs)  
* Explain which rules passed/failed

**Tech Stack**:

* Custom Python DSL or OPA (Open Policy Agent)  
* JSON schema for rule definitions  
* Rule versioning (git-based or DB)

**Key Files**:

agents/rules/  
├── rules\_engine.py           \# Rule evaluator  
├── rule\_dsl.py               \# Parser for rule language  
├── rule\_library/  
│   ├── insurance\_rules.json  
│   ├── healthcare\_rules.json  
│   ├── finance\_rules.json  
│   └── manufacturing\_rules.json  
└── explainer.py              \# Generate human-readable rationales

**Example Rule (Healthcare Prior Auth)**:

{  
  "rule\_id": "PA\_R1\_shoulder\_surgery",  
  "condition": "CPT in \['27447'\] AND diagnosis in \['M17.11'\]",  
  "requires\_evidence": \["failed\_PT\_6weeks", "NSAID\_trial"\],  
  "action": "needs\_evidence"  
}

---

### **2.6 Algorithm Agent**

**Responsibility**: Execute exact, auditable computations (fuzzy matching, graph traversal, temporal analysis)

**Key Operations**:

* String similarity (Jaro-Winkler, TF-IDF cosine, Levenshtein)  
* Temporal windowing (±N days, dynamic time warping)  
* Graph algorithms (shortest path, bipartite matching, max-flow)  
* Financial reconciliation scoring (multi-signal fusion)

**Tech Stack**:

* Python with NumPy/SciPy  
* NetworkX for graph algorithms  
* Fuzzy matching libraries (rapidfuzz, jellyfish)

**Key Files**:

agents/algorithms/  
├── algorithm\_agent.py        \# Main dispatcher  
├── string\_matching.py        \# Fuzzy string algorithms  
├── temporal\_analysis.py      \# Time-series alignment  
├── graph\_algorithms.py       \# Path finding, matching  
└── reconciliation\_scorer.py  \# Multi-signal scoring (GL↔Bank)

**Example: GL↔Bank Reconciliation Score**:

score \= (  
    0.3 \* date\_proximity(gl\_date, bank\_date) \+  
    0.25 \* amount\_similarity(gl\_amount, bank\_amount) \+  
    0.2 \* memo\_cosine\_similarity(gl\_memo, bank\_desc) \+  
    0.15 \* payer\_exact\_match(gl\_payer, bank\_payer) \+  
    0.1 \* swift\_reference\_match(gl\_ref, bank\_swift)  
)

---

### **2.7 ML Model Agent**

**Responsibility**: Provide probabilistic predictions (match likelihood, fraud scores, approval probability)

**Key Operations**:

* Serve pre-trained models (XGBoost, LightGBM, fine-tuned BERT)  
* Batch inference for multiple candidates  
* Return calibrated probabilities with confidence intervals  
* Track model versions and drift

**Tech Stack**:

* scikit-learn, XGBoost, LightGBM  
* Hugging Face transformers (BERT for text classification)  
* MLflow for model registry  
* ONNX Runtime or Triton for serving

**Key Files**:

agents/ml/  
├── ml\_agent.py                  \# Model serving orchestrator  
├── models/  
│   ├── match\_classifier.pkl     \# XGBoost for GL↔Bank matching  
│   ├── fraud\_detector.pkl       \# Anomaly detection  
│   ├── prior\_auth\_predictor.pkl \# Healthcare approval probability  
│   └── subro\_recovery\_model.pkl \# Insurance recovery prediction  
├── feature\_pipeline.py          \# Feature extraction for models  
├── calibration.py               \# Platt/Isotonic scaling  
└── drift\_monitor.py             \# Evidently AI integration

**Critical**: All probabilities must be **calibrated** (use Platt scaling or isotonic regression). Track Brier scores and calibration curves.

---

### **2.8 GenAI Reasoner Agent**

**Responsibility**: Semantic understanding, evidence extraction, cross-document reasoning, narrative generation

**Key Operations**:

* Extract evidence from unstructured text with citations (span offsets)  
* Cross-document entailment ("does Note A satisfy Rule R1?")  
* Generate proposals (subro letter, PA justification, reconciliation explanation)  
* Dialogic clarification with HITL (create checklists for missing info)

**Tech Stack**:

* OpenAI API (GPT-4) or Anthropic Claude  
* LangChain for RAG pipelines  
* Structured output via JSON schema (function calling)  
* Citation tracking (return text spans \+ confidence)

**Key Files**:

agents/genai/  
├── genai\_agent.py              \# LLM orchestrator  
├── evidence\_extractor.py       \# Extract claims with citations  
├── entailment\_checker.py       \# Cross-doc reasoning  
├── proposal\_generator.py       \# Draft letters, justifications  
├── dialogue\_manager.py         \# HITL clarification questions  
└── citation\_validator.py       \# Verify evidence spans exist

**Critical Guardrails**:

* **NO direct writes** to production systems  
* Every claim must cite source spans  
* JSON schema enforcement for all outputs  
* Grounding rate \>98% (outputs with valid evidence)

**Example Prompt (Evidence Extraction)**:

prompt \= f"""  
Extract evidence from the following clinical note that satisfies the prior authorization requirement:  
"Patient must have failed PT for 6+ weeks OR tried NSAID for 6+ weeks"

Clinical Note:  
{note\_text}

Return JSON:  
{{  
  "evidence\_found": bool,  
  "evidence\_type": "failed\_PT" | "NSAID\_trial" | null,  
  "text\_span": "exact quoted text",  
  "span\_offset": \[start\_char, end\_char\],  
  "confidence": 0.0-1.0  
}}  
"""

---

### **2.9 Assurance Agent (Uncertainty Quantification)**

**Responsibility**: Score uncertainty across all modalities, flag low-confidence cases for HITL

**Key Operations**:

* Compute uncertainty from ML models (MC-dropout, ensemble variance)  
* Score LLM faithfulness (evaluate citations, check hallucinations)  
* Aggregate uncertainty: `u = f(ml_variance, llm_confidence, rule_conflicts, alg_noise)`  
* Emit confidence scores for Policy Agent

**Tech Stack**:

* Python with NumPy for uncertainty propagation  
* LLM-as-judge pattern (use GPT-4 to evaluate GPT-3.5 outputs)  
* Ensemble methods for ML uncertainty

**Key Files**:

agents/assurance/  
├── assurance\_agent.py          \# Main uncertainty scorer  
├── ml\_uncertainty.py           \# MC-dropout, ensemble variance  
├── llm\_faithfulness.py         \# Citation validation, hallucination detection  
├── rule\_conflict\_detector.py   \# Flag contradictory rules  
└── uncertainty\_fusion.py       \# Aggregate across modalities

**Uncertainty Formula**:

uncertainty \= (  
    0.4 \* ml\_variance \+  
    0.3 \* (1 \- llm\_confidence) \+  
    0.2 \* rule\_conflict\_score \+  
    0.1 \* alg\_noise  
)

---

### **2.10 Policy Agent (Decision Fusion)**

**Responsibility**: Combine all modality scores into a final decision (auto-resolve, HITL review, request-info)

**Key Operations**:

* Fuse scores: `U(x,a) = α*alg + β*ml + γ*llm + δ*rule - λ*uncertainty`  
* Apply hard constraints (mandatory rule failures override)  
* Apply thresholds: `τ_auto = 0.65`, `τ_review = 0.45`  
* Emit decision \+ explanation object  
* Log for learning loop

**Tech Stack**:

* Python with NumPy  
* Contextual bandits library (vowpal\_wabbit or custom LinUCB)  
* Constraint solver for governance

**Key Files**:

agents/policy/  
├── policy\_agent.py             \# Decision fusion  
├── utility\_function.py         \# U(x,a) computation  
├── threshold\_manager.py        \# Adaptive thresholds  
├── constraint\_checker.py       \# Hard gates (SOX, HIPAA)  
└── explanation\_builder.py      \# Generate audit-ready explanations

**Decision Logic**:

U \= (  
    policy\_weights\['alpha'\] \* alg\_score \+  
    policy\_weights\['beta'\] \* ml\_prob \+  
    policy\_weights\['gamma'\] \* llm\_score \+  
    policy\_weights\['delta'\] \* rule\_numeric \-  
    policy\_weights\['lambda'\] \* uncertainty  
)

if any\_mandatory\_rule\_failed:  
    decision \= "request\_info"  
elif U \>= tau\_auto and constraints\_satisfied:  
    decision \= "auto\_resolve"  
elif U \>= tau\_review:  
    decision \= "hitl\_review"  
else:  
    decision \= "request\_info"

---

### **2.11 Action Agent**

**Responsibility**: Execute approved decisions (write-backs, generate documents, route tasks)

**Key Operations**:

* Write reconciliations to ERP (journal entries)  
* Generate subro letters, PA approval letters  
* Create tickets in case management systems  
* Route cases to human queues with AI briefs  
* All actions are transactional (rollback on failure)

**Tech Stack**:

* Python with asyncio for concurrent writes  
* ERP/CRM SDKs (SAP API, Salesforce REST)  
* Document generation (python-docx, ReportLab for PDFs)  
* Email APIs (SendGrid, Gmail API)

**Key Files**:

agents/action/  
├── action\_agent.py             \# Main action executor  
├── write\_back/  
│   ├── erp\_writer.py           \# GL journal entries  
│   ├── crm\_writer.py           \# Update claim status  
│   └── ticket\_creator.py       \# Jira, ServiceNow  
├── document\_generation/  
│   ├── letter\_generator.py     \# Subro, PA letters  
│   └── pdf\_builder.py          \# Audit packs  
└── transaction\_manager.py      \# Rollback on failure

---

### **2.12 Learning Agent (Continuous Improvement)**

**Responsibility**: Capture feedback, compute rewards, update models and policy weights

**Key Operations**:

* Log every decision: `(context, action, propensity, outcome)`  
* Compute rewards from outcomes (approved/paid, cycle time, reversals)  
* Update policy weights via bandit algorithms (Thompson Sampling, LinUCB)  
* Trigger model retraining when drift detected  
* A/B test policy variants in shadow mode

**Tech Stack**:

* Python with vowpal\_wabbit (contextual bandits)  
* Offline RL libraries (d3rlpy for CQL/AWR)  
* MLflow for experiment tracking  
* Evidently AI for drift monitoring

**Key Files**:

agents/learning/  
├── learning\_agent.py           \# Feedback loop orchestrator  
├── reward\_shaper.py            \# R \= f(accuracy, cycle\_time, risk)  
├── bandit\_updater.py           \# Thompson Sampling / LinUCB  
├── offline\_rl.py               \# CQL for multi-step decisions  
├── drift\_detector.py           \# Evidently integration  
└── policy\_evaluator.py         \# Counterfactual evaluation

**Reward Function**:

reward \= (  
    w\_acc \* int(correct) \-  
    w\_rev \* int(reversed) \+  
    w\_time \* cycle\_time\_saved \-  
    w\_risk \* risk\_score  
)

**Bandit Update** (nightly):

\# Collect logged decisions from past 24h  
contexts, actions, rewards, propensities \= load\_logs()

\# Update policy weights  
policy\_weights \= thompson\_sampling\_update(  
    contexts, actions, rewards,  
    prior=policy\_weights,  
    constraints=governance\_constraints  
)

---

## **3\. Demo Scenarios (Multi-Vertical)**

### **3.1 Insurance: Subrogation Recovery**

**Case**: Claim paid; need to determine if other party is liable and auto-draft subro notice

**Agent Flow**:

1. **Retriever**: Fetch claim data, police report PDF, adjuster notes  
2. **Data**: Extract entities (parties, vehicles, fault statements), build claim graph  
3. **Rules**: Check coverage, timing windows (statute of limitations)  
4. **Algorithm**: Fuzzy match remittance text to claim ID  
5. **ML**: Predict recovery probability  
6. **GenAI**: Extract "other vehicle ran red light" from police report with citation  
7. **Assurance**: Score confidence across modalities  
8. **Policy**: U ≥ 0.70 → auto-resolve  
9. **Action**: Draft subro letter, re-link payment, update claim status  
10. **Learning**: Log decision; if recovery succeeds, reward \= \+$recovery\_amount

**Demo Output**: Side-by-side UI showing:

* Rules panel (coverage ✓, timing ✓)  
* Algorithm score (text match: 0.87)  
* ML probability (recovery: 0.78)  
* LLM evidence (highlighted quote from police report)  
* Final U score: 0.82 → **AUTO-RESOLVED**  
* Generated subro letter with citations

---

### **3.2 Healthcare: Prior Authorization**

**Case**: Provider submits PA request for shoulder surgery; need evidence of failed conservative treatment

**Agent Flow**:

1. **Retriever**: Fetch clinical notes (past 6 months), PA request form  
2. **Data**: NER for CPT codes, diagnosis codes, treatment mentions  
3. **Rules**: Encode payer criteria (CPT 27447 \+ M17.11 requires "failed PT ≥6 weeks")  
4. **GenAI**: Hunt through long clinical notes for "PT trial" or "NSAID trial" mentions  
5. **ML**: Predict approval probability given longitudinal context  
6. **Assurance**: Check if LLM citations are grounded (spans exist in notes)  
7. **Policy**: If evidence found \+ rule satisfied → auto-approve; else → checklist  
8. **Action**: Approve with audit packet OR generate "Missing Evidence" checklist back to provider  
9. **Learning**: Log; if payer later denies, penalize false positive

**Demo Output**:

* Rules panel: "Requires failed PT ≥6 weeks" ❌ (not found)  
* LLM evidence: No citation found  
* Final decision: **REQUEST INFO** → Checklist: "Please provide PT visit dates or NSAID prescription records"

---

### **3.3 Finance: GL↔Bank Reconciliation**

**Case**: Bank statement shows wire transfer; need to match to GL entry

**Agent Flow**:

1. **Retriever**: Pull GL entries, bank statement CSV  
2. **Data**: Normalize dates, amounts, extract memo fields  
3. **Rules**: SOX controls (high-value entries require SWIFT evidence)  
4. **Algorithm**: Multi-signal similarity (date ±1 day, amount Δ, memo cosine, payer match)  
5. **ML**: Classify match vs. exception  
6. **GenAI**: Generate human-readable explanation citing SWIFT fields  
7. **Policy**: alg=0.92, ml=0.88, rule=pass, U=0.86 → **AUTO-RECONCILE**  
8. **Action**: Post reconciliation, stage JE for review  
9. **Learning**: If auditor later flags mismatch, penalize error

**Demo Output**:

* Algorithm score breakdown (date: 1.0, amount: 0.95, memo: 0.85)  
* ML probability: 0.88  
* Rule status: SOX controls ✓ (SWIFT ref present)  
* LLM explanation: "Reconciled GL-20250905-0042 to BNK-20250906-1823 based on SWIFT MT103 reference"  
* **AUTO-RECONCILED**

---

### **3.4 Retail/CPG: Return Authorization**

**Case**: Customer returns bundle product; need to decompose and validate components

**Agent Flow**:

1. **Retriever**: Order history, return request, product catalog  
2. **Data**: Build product graph (bundle ↔ components)  
3. **Algorithm**: Bipartite matching (returned items ↔ original components)  
4. **Rules**: Return window (30 days), abuse detection  
5. **ML**: Fraud/abuse scorer  
6. **Policy**: If all components matched \+ within window \+ low fraud score → approve  
7. **Action**: Issue refund, update inventory  
8. **Learning**: Track fraud hit rate

---

### **3.5 Manufacturing: Batch Traceability (Recall)**

**Case**: Defective component; need to trace all affected products

**Agent Flow**:

1. **Retriever**: Fetch batch logs, vendor PDFs, BOM data  
2. **Data**: OCR batch IDs from vendor docs, build lineage graph  
3. **Rules**: Check spec compliance (tolerances, certifications)  
4. **Algorithm**: Shortest path from defective batch to finished goods  
5. **GenAI**: Extract root cause from vendor quality report  
6. **Policy**: If spec violation \+ affected products identified → raise CAPA  
7. **Action**: Generate recall list, create CAPA ticket  
8. **Learning**: Track recall cost, time to resolution

---

## **4\. Technology Stack (Opinionated)**

### **4.1 Core Infrastructure**

| Component | Choice | Rationale |
| ----- | ----- | ----- |
| **Language** | Python 3.11+ | Mike's preference; rich ML/AI ecosystem |
| **Orchestration** | Temporal.io | Fault-tolerant workflows, better than Airflow for real-time |
| **Message Queue** | Redis (dev), Kafka (prod) | Lightweight for local, scalable for production |
| **Vector DB** | ChromaDB (dev), Weaviate (prod) | Easy local setup, production-grade cloud option |
| **Graph DB** | NetworkX (dev), Neo4j (prod) | Lightweight graph ops, scalable labeled property graph |
| **Feature Store** | Postgres \+ DuckDB | OLTP \+ OLAP in one, avoid over-engineering |
| **ML Serving** | FastAPI \+ ONNX Runtime | Low latency, easy deployment |
| **LLM** | OpenAI GPT-4 or Anthropic Claude | Best-in-class for reasoning \+ function calling |
| **Monitoring** | Evidently AI \+ Grafana | Drift detection \+ observability |

### **4.2 Development Environment**

* **OS**: Windows 11 (Mike's setup)  
* **IDE**: VS Code  
* **Shell**: PowerShell  
* **Containerization**: Docker Desktop  
* **Testing**: pytest, hypothesis (property-based testing)  
* **Linting**: ruff (fast Python linter), mypy (type checking)

---

## **5\. Project Structure**

qure/  
├── agents/  
│   ├── orchestration/  
│   ├── retriever/  
│   ├── data/  
│   ├── rules/  
│   ├── algorithms/  
│   ├── ml/  
│   ├── genai/  
│   ├── assurance/  
│   ├── policy/  
│   ├── action/  
│   └── learning/  
├── substrate/  
│   ├── vector\_store.py  
│   ├── graph\_store.py  
│   ├── feature\_store.py  
│   └── evidence\_tracker.py  
├── common/  
│   ├── schemas/           \# Protobuf or Pydantic models  
│   ├── config/            \# YAML configs per environment  
│   └── utils/  
├── demos/  
│   ├── insurance\_subro/  
│   ├── healthcare\_pa/  
│   ├── finance\_recon/  
│   ├── retail\_returns/  
│   └── manufacturing\_trace/  
├── tests/  
│   ├── unit/  
│   ├── integration/  
│   └── e2e/  
├── ui/  
│   ├── backend/           \# FastAPI  
│   └── frontend/          \# React (optional, or Streamlit)  
├── data/  
│   ├── synthetic/         \# Demo datasets  
│   └── models/            \# Pre-trained ML models  
├── scripts/  
│   ├── setup\_env.sh       \# Environment setup  
│   ├── generate\_data.py   \# Synthetic data generator  
│   └── run\_demo.py        \# End-to-end demo orchestrator  
├── docker/  
│   ├── docker-compose.yml  
│   └── Dockerfile  
├── docs/  
│   ├── architecture.md  
│   ├── agent\_specs.md  
│   └── api\_reference.md  
├── .env.example  
├── pyproject.toml         \# Poetry or pip-tools  
└── README.md

---

## **6\. Implementation Phases (Realistic Timeline)**

### **Phase 1: Foundation (Week 1-2)**

* \[ \] Set up project structure  
* \[ \] Implement Knowledge Substrate (ChromaDB \+ NetworkX)  
* \[ \] Create common schemas (Pydantic models)  
* \[ \] Build synthetic data generator for Finance use case  
* \[ \] Implement Retriever Agent (S3 \+ CSV connectors)  
* \[ \] Implement Data Agent (basic NER, embeddings, graph construction)

### **Phase 2: Reasoning Mesh (Week 3-4)**

* \[ \] Implement Rules Engine Agent (DSL \+ Finance rules)  
* \[ \] Implement Algorithm Agent (reconciliation scoring)  
* \[ \] Implement ML Model Agent (train simple XGBoost classifier)  
* \[ \] Implement GenAI Reasoner Agent (OpenAI integration \+ RAG)  
* \[ \] Implement Assurance Agent (uncertainty scoring)

### **Phase 3: Decision & Action (Week 5-6)**

* \[ \] Implement Policy Agent (decision fusion logic)  
* \[ \] Implement Action Agent (mock write-backs, letter generation)  
* \[ \] Implement Orchestration Agent (Temporal workflows)  
* \[ \] End-to-end integration test (Finance reconciliation)

### **Phase 4: Learning Loop (Week 7\)**

* \[ \] Implement Learning Agent (logging, reward computation)  
* \[ \] Add Thompson Sampling bandit for policy weight updates  
* \[ \] Drift monitoring with Evidently AI  
* \[ \] Counterfactual evaluation harness

### **Phase 5: Multi-Vertical Demo (Week 8\)**

* \[ \] Port to Insurance Subro (new rules, data generator)  
* \[ \] Port to Healthcare PA (new rules, clinical note parser)  
* \[ \] Build demo UI (Streamlit or React)  
* \[ \] Create executive demo script  
* \[ \] Documentation \+ deployment guide

---

## **7\. Demo UI Specification**

### **7.1 Key Screens**

1. **Case Loader**: Upload or select a case (GL entry, claim, PA request)  
2. **Agent Execution View**: Real-time log of agent activations  
3. **Evidence Panel**:  
   * Rules (✓/❌ with rationales)  
   * Algorithm scores (breakdown)  
   * ML probabilities  
   * LLM evidence (highlighted text spans with citations)  
   * Uncertainty score  
4. **Policy Math Tab**: Show U computation, thresholds, decision  
5. **Action Preview**: Draft letter, reconciliation, or checklist  
6. **Learning Dashboard**:  
   * Exception rate trend  
   * Policy weight evolution  
   * Calibration curves  
   * Case resolution distribution (auto/HITL/request-info)

### **7.2 Tech Stack**

* **Backend**: FastAPI (Python)  
* **Frontend**: Streamlit (rapid prototyping) OR React \+ D3.js (polished)  
* **Real-time updates**: Server-Sent Events (SSE) or WebSockets

---

## **8\. Critical Success Factors**

### **8.1 Brutal Honesty Assessment**

**What Can Go Wrong**:

1. **LLM hallucinations**: Mitigate with grounding checks, citation validation  
2. **Model drift**: Monitor with Evidently, retrain weekly  
3. **Rule conflicts**: Implement conflict resolution logic, explicit precedence  
4. **Integration brittleness**: Mock all external APIs, extensive error handling  
5. **Latency**: Optimize with caching (Redis), async I/O, batch inference

**Realistic Expectations**:

* **Week 1-4**: Working prototype on ONE vertical (Finance)  
* **Week 5-6**: Multi-agent orchestration, decision quality at 70-80% precision  
* **Week 7-8**: Multi-vertical demo, learning loop showing improvement  
* **Beyond**: 6-12 months for production hardening

### **8.2 Non-Negotiables**

1. **Every LLM output must have citations** (span offsets to source text)  
2. **All probabilities must be calibrated** (track Brier scores)  
3. **Audit logs are immutable** (append-only, cryptographically signed)  
4. **No direct writes from LLMs** (always mediated by Action Agent)  
5. **HITL by design** (never 100% automation, always confidence-based escalation)

---

## **9\. Competitive Positioning (Key Talking Points)**

**QURE vs. Traditional RPA**:

* RPA: Linear workflows, breaks on exceptions  
* QURE: Contextual reasoning, learns from exceptions

**QURE vs. IBM watsonx Orchestrate**:

* watsonx: General-purpose orchestration, requires custom logic  
* QURE: Pre-tuned for back-office verticals (Insurance, Healthcare, Finance)

**QURE vs. LLM Chatbots**:

* Chatbots: Conversational, no action execution  
* QURE: Autonomous resolution with write-backs, governed by rules \+ ML

**Key Message**:

"QURE closes the last-mile automation gap where rules fail and humans take over: by adding reasoning, context, and learning between your systems."

---

## **10\. Next Steps for Claude Code Agents**

### **Agent 1: Project Scaffolding Agent**

**Task**: Create directory structure, initialize configs, set up Docker Compose

### **Agent 2: Data Generation Agent**

**Task**: Generate synthetic datasets for Finance (GL↔Bank), Insurance (Subro), Healthcare (PA)

### **Agent 3: Knowledge Substrate Agent**

**Task**: Implement vector store, graph store, feature store interfaces

### **Agent 4: Retriever \+ Data Agent**

**Task**: Build data ingestion pipeline, NER, embeddings, graph construction

### **Agent 5: Reasoning Mesh Agent**

**Task**: Implement Rules, Algorithm, ML, GenAI agents with mock models

### **Agent 6: Policy \+ Action Agent**

**Task**: Decision fusion logic, mock write-backs, explanation generation

### **Agent 7: Demo UI Agent**

**Task**: Build Streamlit dashboard showing agent execution \+ evidence panel

### **Agent 8: Documentation Agent**

**Task**: Generate API docs, architecture diagrams, user guides

---

## **11\. Success Metrics (Demo Readiness)**

**Week 4 Checkpoint**:

* \[ \] Finance demo resolves 20 synthetic cases  
* \[ \] Rules/Algorithm/ML/LLM scores visible in UI  
* \[ \] Decision logic works (auto/HITL/request-info)  
* \[ \] At least 60% auto-resolution rate  
* \[ \] No hallucinations (100% citation grounding)

**Week 8 Checkpoint**:

* \[ \] Three verticals working (Finance, Insurance, Healthcare)  
* \[ \] Learning loop shows exception rate drop (5-10% improvement)  
* \[ \] Executive demo script under 10 minutes  
* \[ \] Docker one-command deploy  
* \[ \] Documentation complete

---

## **12\. Risk Mitigation**

| Risk | Mitigation |
| ----- | ----- |
| LLM API rate limits | Cache embeddings, use local models (llama.cpp) as fallback |
| Model overfitting on synthetic data | Use domain-expert-reviewed validation set, track calibration |
| Agent coordination deadlocks | Timeout all agent calls (5-30s), circuit breakers |
| Demo data realism | Partner with domain expert (insurance adjuster, accountant) for review |
| Scope creep | Lock verticals at 3 for demo, no new features after Week 6 |

---

## **Appendix A: Key Python Libraries**

\[tool.poetry.dependencies\]  
python \= "^3.11"

\# Core  
fastapi \= "^0.104.0"  
uvicorn \= "^0.24.0"  
pydantic \= "^2.4.0"  
python-dotenv \= "^1.0.0"

\# Orchestration  
temporal-sdk \= "^1.2.0"  \# or prefect, airflow

\# Data Processing  
pandas \= "^2.1.0"  
numpy \= "^1.25.0"  
spacy \= "^3.7.0"  
transformers \= "^4.35.0"

\# ML  
scikit-learn \= "^1.3.0"  
xgboost \= "^2.0.0"  
lightgbm \= "^4.1.0"

\# LLM  
openai \= "^1.3.0"  
langchain \= "^0.0.335"  
tiktoken \= "^0.5.0"

\# Vector/Graph  
chromadb \= "^0.4.15"  
neo4j \= "^5.14.0"  
networkx \= "^3.2.0"

\# Monitoring  
evidently \= "^0.4.0"  
mlflow \= "^2.8.0"

\# Utils  
redis \= "^5.0.0"  
aiohttp \= "^3.9.0"  
pytest \= "^7.4.0"  
pytest-asyncio \= "^0.21.0"

---

## **Appendix B: Environment Variables**

\# .env.example

\# LLM APIs  
OPENAI\_API\_KEY=sk-...  
ANTHROPIC\_API\_KEY=sk-ant-...

\# Vector DB  
CHROMA\_HOST=localhost  
CHROMA\_PORT=8000

\# Graph DB  
NEO4J\_URI=bolt://localhost:7687  
NEO4J\_USER=neo4j  
NEO4J\_PASSWORD=password

\# Redis  
REDIS\_HOST=localhost  
REDIS\_PORT=6379

\# Temporal  
TEMPORAL\_HOST=localhost:7233

\# Feature Flags  
ENABLE\_LEARNING\_LOOP=true  
ENABLE\_BANDIT\_UPDATES=false  \# Start false, enable in Phase 4  
LLM\_PROVIDER=openai  \# or anthropic

\# Demo Settings  
DEMO\_VERTICAL=finance  \# finance|insurance|healthcare  
AUTO\_RESOLVE\_THRESHOLD=0.65  
HITL\_REVIEW\_THRESHOLD=0.45

---

## 