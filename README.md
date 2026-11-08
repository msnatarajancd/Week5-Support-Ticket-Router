# Week 5: Support Ticket Router

Status: notebook prepared and locally tested; GPU training and evaluation have not yet run.

This project follows The Gen Academy's Week 5 assignment: fine-tune Qwen/Qwen3-1.7B-Base through LLaMA Board using LoRA, merge the adapter, and compare routing performance against the base model.

## Open the notebook

[Colab notebook](https://colab.research.google.com/drive/1AvJx4cNOX04fiUEPtxZOhEs9KTX7GXV_)

[Project Drive folder](https://drive.google.com/drive/folders/1eReAPODspet6nHwzFnZNI_LNwjAuRf3I)

Use a T4 runtime, run cells sequentially, and follow the LLaMA Board instructions in phase 3. The notebook prints the pinned model path, training settings, and adapter output directory. Wait for training completion before phase 4. Preserve the executed notebook, evidence ZIP, adapter and any needed merged checkpoint before disconnecting.

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

See artifacts/data/data_audit.json for row IDs and dataset checksum. No model scores are available until the GPU run completes.

## Corrections to the starter notebook

- Explicit class ordering prevents metrics being assigned to the wrong class names.
- Invalid generated labels remain errors instead of silently becoming Support general.
- Data cleaning and held-out overlap checks precede training.
- The pinned base model and training tokenizer/template are reused for inference.
- All seven routing labels have smoke-test coverage; adapter and merged predictions are compared.
- Completion checks reject missing or unfinished training state.
- Exported predictions allow metric recomputation and include raw model output.

Baseline letter-choice prediction and fine-tuned label generation use different decoding protocols, so their accuracy delta compares the complete approaches. First-token scores are not treated as calibrated confidence. No deployment cost or latency benefit has been measured.

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
