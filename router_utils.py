"""Data integrity and evaluation for the Week 5 routing experiment."""
import hashlib
import json
import math
import shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = ('Support general', 'Fileservice', 'O365', 'EOL', 'Software', 'Active Directory', 'Computer-Services')
INVALID = '__invalid__'
SYSTEM_PROMPT = ('You are an IT helpdesk ticket routing assistant. '
                 'Given a support ticket, respond with exactly one of the following categories: '
                 + ', '.join(LABELS) + '.')

def prepare_data(csv_path: str, output_dir: str, seed: int = 42) -> dict:
    source = Path(csv_path)
    df = pd.read_csv(source)
    if not {'text', 'category_truth'}.issubset(df.columns):
        raise ValueError('CSV requires text and category_truth columns')
    df = df.rename(columns={'category_truth': 'label'})[['text', 'label']].copy()
    df['row_id'] = range(len(df))
    audit = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'input_rows': len(df), 'seed': seed}
    empty = df.text.isna() | df.text.fillna('').astype(str).str.strip().eq('')
    audit['empty_rows_removed'] = int(empty.sum())
    df = df.loc[~empty].copy()
    unknown = ~df.label.isin(LABELS)
    audit['unknown_label_rows_removed'] = int(unknown.sum())
    df = df.loc[~unknown].copy()
    df['key'] = df.text.astype(str).str.casefold().str.split().str.join(' ')
    conflict = df.groupby('key').label.transform('nunique').gt(1)
    audit['conflicting_rows_removed'] = int(conflict.sum())
    df = df.loc[~conflict].copy()
    audit['duplicate_rows_removed'] = int(df.duplicated('key').sum())
    df = df.drop_duplicates('key')
    counts = df.label.value_counts()
    if set(counts.index) != set(LABELS) or counts.min() < 2 or math.ceil(len(df)*.2) < 7:
        raise ValueError('Need all seven classes with enough unique examples for a stratified 80/20 split')
    train, val = train_test_split(df, test_size=.2, stratify=df.label, random_state=seed)
    if set(train.key) & set(val.key):
        raise ValueError('Train/validation leakage detected')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = [{'messages': [{'role':'system','content':SYSTEM_PROMPT},
                             {'role':'user','content':'Support Ticket: '+str(row.text)},
                             {'role':'assistant','content':row.label}]}
               for row in train.itertuples()]
    (root/'train.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    val.drop(columns='key').to_csv(root/'val_split.csv', index=False)
    train.drop(columns='key').to_csv(root/'train_split.csv', index=False)
    audit.update(clean_rows=len(df), train_rows=len(train), validation_rows=len(val),
                 class_counts={k:int(v) for k,v in counts.items()},
                 train_class_counts={k:int(v) for k,v in train.label.value_counts().items()},
                 validation_class_counts={k:int(v) for k,v in val.label.value_counts().items()},
                 train_row_ids=train.row_id.tolist(), validation_row_ids=val.row_id.tolist(),
                 normalized_overlap=0)
    (root/'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit

def parse_label(text: str) -> str:
    return next((label for label in LABELS if label.casefold() == text.strip().casefold()), INVALID)

def evaluate_predictions(truth: list[str], predictions: list[str]) -> dict:
    if not truth or len(truth) != len(predictions) or not set(truth).issubset(LABELS):
        raise ValueError('Nonempty equally sized predictions and valid truth labels are required')
    pred = [p if p in LABELS else INVALID for p in predictions]
    report = classification_report(truth, pred, labels=list(LABELS), target_names=list(LABELS),
                                   output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(truth, pred)),
            'macro_f1': report['macro avg']['f1-score'], 'invalid_count':pred.count(INVALID),
            'per_class': {label:report[label] for label in LABELS},
            'confusion_labels': list(LABELS)+[INVALID],
            'confusion_counts': confusion_matrix(truth,pred,labels=list(LABELS)+[INVALID]).tolist()}

def generation_options(tokenizer, template) -> dict:
    return {'eos_token_id': template.get_stop_token_ids(tokenizer),
            'pad_token_id': tokenizer.eos_token_id, 'do_sample': False, 'max_new_tokens': 24}

def verify_merge(adapter_outputs: list, merged_outputs: list, output_path: str) -> dict:
    result = {'adapter_outputs': adapter_outputs, 'merged_outputs': merged_outputs,
              'matches': bool(adapter_outputs) and adapter_outputs == merged_outputs}
    Path(output_path).write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not result['matches']:
        raise ValueError('Merge output mismatch; raw evidence saved for investigation')
    return result

def validate_training_config(actual: dict, expected: dict) -> dict:
    critical = ('model_name_or_path','template','dataset','cutoff_len','finetuning_type')
    mismatches = [k for k in critical if actual.get(k) != expected.get(k)]
    if mismatches:
        raise ValueError('Training configuration mismatch or missing: '+', '.join(mismatches))
    return actual

def preserve_directory(source: str, destination: str) -> list[dict]:
    src, dst = Path(source).resolve(), Path(destination).resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError('Preservation source and destination must be separate directories')
    files = [p for p in src.rglob('*') if p.is_file()]
    if not files:
        raise ValueError('Nothing to preserve')
    manifest = []
    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream,'sha256').hexdigest()
    for path in files:
        target=dst/path.relative_to(src)
        target.parent.mkdir(parents=True,exist_ok=True)
        expected=digest(path)
        shutil.copy2(path,target)
        if digest(target) != expected or target.stat().st_size != path.stat().st_size:
            raise ValueError('Preserved file verification failed: '+str(target))
        manifest.append({'path':str(path.relative_to(src)),'bytes':target.stat().st_size,'sha256':expected})
    return manifest
