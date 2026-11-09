# Week 5: Support Ticket Router

Status: completed November 8, 2026. The final T4 run completed all 177 steps; evaluation, merging and verified Drive preservation are complete. The runtime has been disconnected and deleted with user approval.

## Results and deliverables

| Measure | Baseline | Fine-tuned |
|---|---:|---:|
| Held-out accuracy | 35.90% (42/117) | 84.62% (99/117) |
| Macro F1 | 0.3004 | 0.8272 |
| Invalid labels | 0 | 0 |

Accuracy improved by 48.72 percentage points. Smoke tests were 6/7 correct, with all seven outputs valid; the account-creation example was incorrectly routed to Support general. Adapter and merged-model raw outputs matched on all seven smoke cases. The held-out set was evaluated once after the termination correction.

- [Completed Google Doc report](https://docs.google.com/document/d/1Cq5_-cmUo6GTmAvCjbjN4XSvDwwUBiGJg8dT86jHcKc/edit), with four authentic figures; PDF copy: `artifacts/Week5_Project_Report.pdf`.
- [Preserved model and evidence](https://drive.google.com/drive/folders/1A-web_Oehc26cDI0lYCik93lsb-excPf), inside the requested Week 5 project folder.
- Clean reproduction notebook: `notebooks/Week5_Support_Ticket_Router.ipynb`; executed evidence: `artifacts/Week5_Support_Ticket_Router_Executed.ipynb`.
- Metrics, predictions, configurations, training logs, preservation hashes and the actual Colab screenshot are in `artifacts/`.

Fifteen regression tests passed. Independent review found no actionable issues in the final execution evidence or compatibility changes. The reviewer recomputed results and checked manifest consistency and evidence hashes; it did not separately download and rehash the multi-GiB weights. The runtime verified source/destination hashes before shutdown. All five PDF pages were visually checked.

This project follows The Gen Academy's Week 5 assignment: fine-tune Qwen/Qwen3-1.7B-Base through LLaMA Board using LoRA, merge the adapter, and compare routing performance against the base model.

## Open the notebook

[Colab notebook](https://colab.research.google.com/drive/1hX7lF46sAjOaq-Jbhth56AHEQHYTFXLm)

[Project Drive folder](https://drive.google.com/drive/folders/1eReAPODspet6nHwzFnZNI_LNwjAuRf3I)

Use a T4 runtime, mount Drive, run cells sequentially, and follow the LLaMA Board instructions in phase 3. The notebook prints the pinned model path, training settings, and persistent adapter output directory. After starting training, phase 4 waits for a verified final checkpoint; the remaining cells can be queued behind it. They evaluate, merge and preserve outputs with SHA-256 checks. The first setup cell saves live evidence and training checkpoints directly to Drive so a runtime reset does not discard them.

The notebook includes its helper code and downloads the official dataset from a pinned source revision. It does not require this repository to be published.

## Verified data audit

The official CSV contains 585 rows. Validation found no empty tickets, unknown labels, normalized duplicates or conflicting labels. The reproducible 80/20 split contains 468 training rows and 117 validation rows. Normalized text overlap is zero.

| Label | Train | Validation |
|---|---:|---:|
| Support general | 120 | 30 |
| Fileservice | 110 | 28 |
| O365 | 74 | 18 |
| EOL | 46 | 12 |
| Software | 46 | 12 |
| Active Directory | 39 | 9 |
| Computer-Services | 33 | 8 |

See artifacts/data/data_audit.json for row IDs and dataset checksum. Saved predictions and metrics provide the measured results above.

## Corrections to the starter notebook

- Explicit class ordering prevents metrics being assigned to the wrong class names.
- Invalid generated labels remain errors instead of silently becoming Support general.
- Data cleaning and held-out overlap checks precede training.
- The pinned base model and training tokenizer/template are reused for inference.
- All seven routing labels have smoke-test coverage; adapter and merged predictions are compared.
- Completion checks reject missing or unfinished training state.
- Exported predictions allow metric recomputation and include raw model output.
- The base-model LoRA run retains its native `<|endoftext|>` token using LLaMA-Factory's `default` template. The initial `qwen3_nothink` run produced category prefixes followed by unwanted continuation; its raw evidence is retained in `artifacts/attempt1_chat_eos`. The retry changes the template before any fine-tuned held-out evaluation, without relaxing label parsing.
- Token lengths are calculated by encoding formatted text explicitly, avoiding a Transformers 5 return-type change.

The final preservation cell copies the adapter, merged model, tokenizer and evidence into Google Drive and checks every copied file's size and SHA-256. Large weights stay outside Git. LLaMA Board is accessed through Colab's authenticated port iframe. The [earlier diagnostic notebook](https://colab.research.google.com/drive/1AvJx4cNOX04fiUEPtxZOhEs9KTX7GXV_) records the initial output-format failure and a later runtime interruption; it is not the final deliverable.

Baseline letter-choice prediction and fine-tuned label generation use different decoding protocols, so their accuracy delta compares the complete approaches. First-token scores are not treated as calibrated confidence. No deployment cost or latency benefit has been measured.

This is a single seeded holdout with only 8–30 validation tickets per class. Nine of 18 errors route Support general to Software; Software precision is 50%, and Computer-Services recall is 62.5%. Review label boundaries, collect a separate test set and use human review before production deployment. An intermediate corrected run lost its temporary runtime; the successful run restarted from the base model with checkpoints and live evidence written directly to Drive. Earlier attempts remain available for audit.

## Local validation

Python 3.12 or a compatible modern Python environment:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-local.txt
.venv/Scripts/python.exe scripts/build_notebook.py
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The tests cover split integrity, duplicate/conflict handling, sparse classes, metric labels, malformed output handling, notebook compilation and execution of its embedded data preparation helper. Local tests do not verify GPU training or dependency compatibility on Colab.

## Sources and licensing

The unchanged starter notebook and provenance are in original/. The official source is https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router. Course materials are retained for this assignment; this repository does not grant a new license to third-party data or model weights. Inspect source/model licenses before redistributing.

Model weights, credentials and environments are excluded from Git. This project has not been published to GitHub or submitted to the course form.
