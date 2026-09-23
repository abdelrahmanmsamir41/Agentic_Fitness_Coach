# Agentic-Fitness-Trainer

A B2B SaaS knowledge-ingestion pipeline for gym coaches: a coach uploads a client's training-plan PDF, and it's parsed, cleaned, structured, cross-referenced against a canonical exercise database, and stored as a queryable, versioned knowledge base per client — ready for a retrieval-augmented chatbot to answer questions grounded in what the coach actually wrote.

This README documents the **ingestion pipeline** as it stands today (the query/chat side is not yet built — see [Status](#status)).

## Status

**Built and working:**
- End-to-end PDF ingestion: OCR → structured LLM extraction → day-number resolution → canonical exercise-database enrichment → chunking → incremental storage in MongoDB + Qdrant.
- Bilingual (Arabic/English) content handling throughout.
- Incremental, per-document sync to both stores — re-ingesting one document never touches any other coach's or client's data, and unchanged content is never re-embedded.
- A zero-downtime, verified rebuild path for Qdrant (`reindex()`) for the rare case of a full-corpus rebuild (e.g. changing embedding models).

**Not yet built:**
- The query-time chat/retrieval pipeline itself (`retrieve_demo.py` is a manual vector-search smoke test, not a chat interface).
- A two-hop query resolution for "exercise on day N" style questions (day→page linkage already exists in the data; nothing consumes it at query time yet).
- Video-transcript enrichment (each exercise carries a `video_url`, currently unused beyond storage).

## Flow: from PDF upload to a queryable knowledge base

```mermaid
flowchart TD
    A["Coach uploads a training-plan PDF"] --> B["autocrop.cropped_copy()\n(trim scan margins before OCR)"]

    B --> C1["PaddleOCR-VL\n(cloud OCR, whole-PDF job,\n~30s/page)"]
    B --> C2["PyMuPDF: extract_links()\n(hyperlinks per page,\ntop-to-bottom reading order)"]
    B --> C3["PyMuPDF: extract_native_text()\n(the PDF's own embedded text layer,\nif any)"]
    C1 & C2 & C3 -. run concurrently .-> D

    D["native_text.prefer_native_when_ocr_diverges()\n+ append_missing_native_numbers()\n(catches OCR misreading background/graphic\ntext over an invisible native-text layer)"] --> E["flag_unreliable_pages()\n(warn if a page has neither a\nnative-text fallback nor trustworthy OCR)"]
    E --> F["links.merge_links_into_pages()\n(append a '_Links:' block per page,\nurls kept separate from table cells)"]
    F --> G[("pages.json cache\n(ingestion_output/&lt;doc_id&gt;/)")]

    G --> H["chunks.split into pages\n(one page = one unit, never merged)"]
    H --> I["Batch pages, 4 per call\n(_batch)"]
    I --> J["Gemini 2.5 Flash\nwith_structured_output(TrainingBatchResult)\n(TRAINING_BATCH_PROMPT: page_topic + category\nclassification, structured entity extraction,\nblank-cell='0' rule, video-link placement,\nOCR-quirk handling -- no text reconstruction)"]
    J --> K["_match_batch_results()\nmatch each result back to its page\nby page_index (order isn't guaranteed);\ncleaned_text = that page's own preprocessed\nOCR markdown (pages.json), not model output"]
    K --> O[("extractions.json cache\n(one TrainingSectionResult per page)")]

    O --> P["enrich.resolve_day_numbers()\nsecond, cheap Gemini call: page-topics only\n(not full text), matches each\nweight_exercise/cardio page to a\nDay N from the split-table page,\ncross-lingual (Arabic↔English topic phrasing)"]
    P --> Q["day_number stamped onto\nevery weight_exercise/cardio\npage's extraction"]

    Q --> R["enrich.resolve_exercise_reference()\n-> exercise_reference.match_exercises()\n(local, no LLM)"]
    R --> R1["rules.py: gazetteer keyword rules\n(word-boundary regex; 'Cable lat pullover'\n→ equipment=Cable, bodypart=Lats;\nalways wins when it fires)"]
    R --> R2["embeddings.py + matching.py:\nBGE-m3 cosine top-5 candidates\n→ RapidFuzz re-rank\n→ accept only ≥0.72 combined score"]
    R1 & R2 --> S["ref_type / ref_bodypart / ref_equipment /\nref_level / ref_match_score\nstamped onto each weight_exercise entity"]

    S --> T["chunks.build_all()\nfor every page:\n1 parent record (page-level context)\n+ N leaf records (one per structured\nentity: exercise/cardio/glossary/advice/\nsection-fallback), each with a\nsynthesized embed_text string"]

    T --> U["store_mongo.sync_document()\ndelete this document_id's rows,\nre-insert manifest + parents + entities\n(other documents untouched)"]
    T --> V["store_qdrant.sync_document()\ncontent-hash diff per chunk:\nunchanged → skip re-embedding\nchanged/new → BGE-m3 embed + upsert\nno-longer-produced → delete by point id"]

    U --> W[("MongoDB\ndocuments / parents / entities")]
    V --> X[("Qdrant\ncyt_chunks alias → live generation\n(leaf payload + BGE-m3 vector)")]

    W -. provenance and full-page context .-> Y["Ready for retrieval\n(store_qdrant.search(), filtered by\nclient_id/active/chunk_type/day_number)"]
    X -. vector similarity search .-> Y
```

## Pipeline stages in detail

### 1. Parse (`parse.py`, `ocr/`, `links.py`)
- `autocrop.cropped_copy()` trims scan margins before sending the PDF to OCR.
- Three independent, concurrently-run extractions: **PaddleOCR-VL** (cloud, whole-PDF job submission + polling, the slow ~30s/page step), **PyMuPDF link extraction** (hyperlinks, sorted top-to-bottom/left-to-right to match reading order), and **PyMuPDF native-text extraction** (the PDF's own embedded text layer, when present).
- `native_text.py` reconciles OCR against native text: if OCR's output for a page is missing numbers that the native layer has, that's treated as a sharp, false-positive-resistant signal that OCR silently dropped real data (e.g. misreading a background graphic instead of an invisible native-text table) — the missing numbers get appended back in. `flag_unreliable_pages()` warns when a page has neither signal to trust.
- `ocr/tables.py` compacts raw HTML `<table>` markup from PaddleOCR-VL into a token-lean custom format (`TABLE[type=data]`/`TABLE[type=mapping]`, columns line + pipe-separated rows) before anything reaches an LLM.
- `links.merge_links_into_pages()` appends a `_Links:` block per page rather than inlining URLs into table cells — this is what lets the extraction prompt tell the model to reconstruct full markdown tables with a video-link column, and keeps table cells faithful to the raw OCR output (blank stays blank).
- Result is cached to `ingestion_output/<doc_id>/pages.json`; a re-run reuses it instead of re-hitting the OCR API.

### 2. Structured extraction (`extraction/extract.py`, `extraction/models.py`, `extraction/prompts.py`)
- Pages are batched **4 per Gemini call** (`PAGES_PER_LLM_CALL`), not one call per page — cuts LLM call count roughly 5x. Each page is tagged with `page_index` in the prompt so the model's response can be matched back to its source page even if it reorders or drops one.
- Schema (`TrainingSectionResult`, Pydantic + `with_structured_output`): a `category` (`weight_exercise` / `cardio` / `glossary` / `advice` / `program_structure` / `marketing` / `other`) plus category-specific entity lists — the model only classifies and extracts structured fields, it doesn't reconstruct page text.
- `cleaned_text` is **not model output**: it's set directly from that page's own preprocessed OCR markdown (`ocr/preprocess.py`'s `strip_noise` output, i.e. what's already in `pages.json`), which turned out to need no further LLM cleanup on top of it. It still flows through the rest of the pipeline (parent record text, fallback "section" leaves) exactly as before.
- `NumberRange {low, high}` replaces loose strings for sets/reps/RIR/speed/incline — a single number becomes `{low: N}`, a range becomes `{low, high}`, a non-numeric value ("AMRAP", "to failure") becomes `null`, and a genuinely blank table cell becomes `{low: 0}` (the model is taught the difference between "no value written" and "not applicable").
- Both the raw OCR pages and the LLM extraction output are cached (`extractions.json`) so a re-run skips Gemini entirely if nothing changed.

### 3. Enrichment (`enrich.py`, `exercise_reference/`)
Two independent post-extraction passes over one document's `all_extractions`, run by `enrich.enrich_document()` after `extract_document()` returns:
- **Day-number resolution** (`enrich.resolve_day_numbers` → `assign_day_numbers`): a second, much cheaper Gemini call — page *topics* only, not full page text — matches every `weight_exercise`/`cardio` page to a `Day N` from the plan's split-table page, handling cross-lingual topic phrasing (e.g. an Arabic page topic matching an English day label). Pages that don't clearly belong to any day are left unassigned rather than guessed.
- **Canonical exercise enrichment** (`enrich.resolve_exercise_reference` → `exercise_reference/`): purely local — **no LLM call**. Every extracted exercise name is resolved against a static reference set (2,909 deduplicated rows from the public `megaGymDataset`) to attach canonical `type`/`bodypart`/`equipment`/`level` fields, via a two-layer funnel:
  1. **Gazetteer keyword rules** (`rules.py`): word-boundary regex over the exercise name ("Cable lat pullover" → `equipment=Cable`, `bodypart=Lats`). Near-certain precision when a rule fires, and always wins over whatever the similarity layer would have said for that field.
  2. **Hybrid similarity match** (`embeddings.py` + `matching.py`): BGE-m3 cosine similarity (same embedding model used for the vector store) proposes the top-5 candidates, re-ranked by a RapidFuzz string score; accepted only above a 0.72 combined threshold, otherwise nothing is attached rather than risking a wrong guess.
  
  Reference embeddings are computed once and cached to disk (`exercise_reference/data/exercises_embeddings.npy`, gitignored) — never recomputed on a normal ingestion run.

### 4. Chunking (`chunks/`)
- One **parent** record per page (full-page context, for retrieval-time expansion) and one **leaf** record per structured entity (exercise, cardio entry, glossary term, advice item), each carrying a synthesized `embed_text` string built from its own fields (not from the page's full prose) plus a `day_number`-aware prefix ("Day 1 - Chest & Back > Incline chest press: ...").
- A page with no structured entities (prose-only, category `other`) falls back to one generic "section" leaf built from `cleaned_text`. A `marketing` page and the `program_structure` page itself produce no leaves at all — the latter's day/target mapping only ever exists to stamp `day_number` onto other pages.

### 5. Storage — incremental, isolated per document (`store_mongo.py`, `store_qdrant.py`)
- **MongoDB**: `sync_document()` deletes and re-inserts only this `document_id`'s rows across `documents`/`parents`/`entities` — no other document is ever touched.
- **Qdrant**: `sync_document()` diffs by a content hash of each chunk's `embed_text` against what's already stored for that document — unchanged chunks are never re-embedded or re-upserted, changed/new ones are, and chunks no longer produced (e.g. an exercise removed from a plan) are deleted by point id.
- **Zero-downtime full rebuild path** (`reindex()`, used only for rare full-corpus operations like switching embedding models): builds into a freshly-named collection, verifies an exact point count, then atomically flips the `cyt_chunks` alias onto it — `search()` always reads through that alias, so it never sees a missing or half-built collection, and a failed build never touches what's live. The previous generation is kept as an instant manual rollback.

## Project structure

```
ingestion/
├── ingest.py                  # orchestrates one document end-to-end; main() loops DOCUMENTS
├── parse.py                   # OCR + links + native-text merge
├── links.py                   # PyMuPDF hyperlink extraction/merge
├── enrich.py                  # post-extraction: day-number resolution + exercise-reference matching
├── lazy.py                    # lazy-singleton helper (shared by store_qdrant/extract/embeddings)
├── text_utils.py              # shared regex helpers (NUMBER, word-boundary pattern builder)
├── ocr/
│   ├── paddle_api.py          # PaddleOCR-VL cloud job submission/polling
│   ├── autocrop.py            # trim scan margins before OCR
│   ├── native_text.py         # native-text vs OCR reconciliation
│   ├── preprocess.py          # OCR text cleanup (abbreviation normalization, etc.)
│   └── tables.py              # raw HTML tables -> compact TABLE[...] text
├── extraction/
│   ├── extract.py             # batched Gemini calls: category classification + structured entity extraction
│   ├── models.py              # Pydantic schemas (TrainingSectionResult, NumberRange, ...)
│   └── prompts.py             # TRAINING_BATCH_PROMPT, DAY_ASSIGNMENT_PROMPT
├── exercise_reference/
│   ├── rules.py                # gazetteer keyword rules
│   ├── embeddings.py           # BGE-m3 loading + cached reference embeddings
│   ├── matching.py             # similarity funnel + rule/match merge
│   └── data/exercises.json     # 2,909-row reference set (megaGymDataset)
├── chunks/
│   ├── build.py                # per-page parent + leaf assembly
│   └── leaves.py               # category-specific leaf builders, embed_text synthesis
├── store_mongo.py             # incremental per-document Mongo sync
├── store_qdrant.py            # incremental per-document Qdrant sync + zero-downtime reindex
└── retrieve_demo.py           # manual vector-search smoke test (not a chat interface)
```

## Running it

Requires `.env` with `GOOGLE_API_KEY`, `PADDLEOCR_TOKEN`, `QDRANT_URL`, `QDRANT_API_KEY`, and `MONGODB_URL` (defaults to `mongodb://localhost:27017/`).

```bash
cd ingestion
python ingest.py
```

Edit `DOCUMENTS` in `ingest.py` to point at the PDF(s) to ingest. Re-running is safe and incremental — cached OCR/extraction output is reused, and only changed content triggers new embedding calls.

To manually check what a query retrieves:

```bash
python retrieve_demo.py "what are my chest exercises"
python retrieve_demo.py --chunk-type weight_exercise,cardio "list my leg exercises"
```
