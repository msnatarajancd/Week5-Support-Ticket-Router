"""Build a standalone Colab notebook; no local project upload is needed."""
import json
import textwrap
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
revision = json.loads((ROOT/'original/provenance.json').read_text(encoding='utf-8-sig'))['source_revision']
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
def code(text, name):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip(), metadata={'name':name}))

md('''# Week 5 — Fine-tune a Support Ticket Router
Prepared for Meenakshi Sundaram Natarajan. Run sequentially on a **T4** GPU.
This notebook follows the official six phases and trains through **LLaMA Board**.
No results are claimed until the cells execute. The baseline uses constrained A–G
choices; the tuned model generates exact labels. The comparison therefore includes
both training and decoding differences.

Source: https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router

Use Runtime → Change runtime type → T4 GPU. Do not purchase compute for this project.
''')
md('## 1. Install dependencies and check GPU')
code('''
import os, sys, subprocess, json, hashlib, shutil, time
from pathlib import Path
import torch
assert torch.cuda.is_available(), 'Select a T4 GPU runtime before continuing.'
GPU = torch.cuda.get_device_name(0)
assert 'T4' in GPU, f'This approved run requires a T4; allocated {GPU}'
ROOT = Path('/content/week5-router')
ROOT.mkdir(exist_ok=True)
ART = ROOT/'artifacts'; ART.mkdir(exist_ok=True)
FACTORY = Path('/content/LLaMA-Factory')
if not FACTORY.exists():
    subprocess.run(['git','clone','--depth','1','https://github.com/hiyouga/LLaMA-Factory.git',str(FACTORY)],check=True)
FACTORY_REV = subprocess.check_output(['git','-C',str(FACTORY),'rev-parse','HEAD'],text=True).strip()
subprocess.run([sys.executable,'-m','pip','install','-q','-e',str(FACTORY)+'[torch,bitsandbytes]','scikit-learn','seaborn'],check=True)
print('GPU:', GPU, 'LLaMA-Factory:', FACTORY_REV)
''', 'install')
md('## 2. Prepare the support-ticket dataset\nExact normalized duplicates are removed and conflicting labels excluded before the seeded split. All exclusions are recorded.')
code((ROOT/'router_utils.py').read_text(encoding='utf-8'), 'helper')
code(f'''
import urllib.request
SOURCE_REV = {revision!r}
CSV = ROOT/'support_tickets.csv'
urllib.request.urlretrieve('https://raw.githubusercontent.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router/'+SOURCE_REV+'/support_tickets.csv', CSV)
assert hashlib.sha256(CSV.read_bytes()).hexdigest() == 'c55c98b462f1af9b2f5e4ad8a7460a6b598e55bcfa42da1b57dc9e921e82783f', 'Dataset checksum changed'
audit = prepare_data(str(CSV),str(ART/'data'))
shutil.copy2(ART/'data/train.json', FACTORY/'data/TRAIN.json')
info_path = FACTORY/'data/dataset_info.json'
info = json.loads(info_path.read_text())
info['support_tickets'] = {{'file_name':'TRAIN.json','formatting':'sharegpt',
    'columns':{{'messages':'messages'}},'tags':{{'role_tag':'role','content_tag':'content',
    'user_tag':'user','assistant_tag':'assistant','system_tag':'system'}}}}
info_path.write_text(json.dumps(info,indent=2))
print(json.dumps({{k:v for k,v in audit.items() if not k.endswith('row_ids')}},indent=2))
''', 'data')
code('''
import sys
sys.path.insert(0, str(FACTORY/'src'))
from transformers import AutoTokenizer
from huggingface_hub import HfApi, snapshot_download
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.hparams import DataArguments
BASE_MODEL = 'Qwen/Qwen3-1.7B-Base'
MODEL_REV = HfApi().model_info(BASE_MODEL).sha
BASE_PATH = snapshot_download(BASE_MODEL,revision=MODEL_REV)
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
template = get_template_and_fix_tokenizer(tokenizer,DataArguments(template='qwen3_nothink'))
TOKENIZER_DIR = ROOT/'training_tokenizer'; tokenizer.save_pretrained(TOKENIZER_DIR)
records = json.loads((ART/'data/train.json').read_text())
max_length = max(len(tokenizer.apply_chat_template(x['messages'],tokenize=True)) for x in records)
CUTOFF = max(1024, ((max_length+63)//64)*64)
assert CUTOFF <= 4096, 'Unexpectedly long data: inspect before training'
ADAPTER_DIR = FACTORY/'saves/Week5-Qwen3-1.7B/lora/router'
MERGED_DIR = ROOT/'merged_model'
CONFIG = {'model_name_or_path':BASE_PATH,'stage':'sft','do_train':True,
    'finetuning_type':'lora','lora_rank':8,'lora_alpha':16,'lora_target':'all',
    'dataset':'support_tickets','dataset_dir':str(FACTORY/'data'),
    'template':'qwen3_nothink','cutoff_len':CUTOFF,'max_samples':100000,
    'output_dir':str(ADAPTER_DIR),'logging_steps':5,'save_steps':100,
    'plot_loss':True,'report_to':'none','per_device_train_batch_size':1,
    'gradient_accumulation_steps':8,'learning_rate':1e-4,'num_train_epochs':3.0,
    'lr_scheduler_type':'cosine','warmup_ratio':.1,'fp16':True,'bf16':False,
    'gradient_checkpointing':True,'seed':42,'overwrite_output_dir':False}
import yaml
(ART/'recommended_training.yaml').write_text(yaml.safe_dump(CONFIG,sort_keys=False))
manifest = {'status':'prepared_not_trained','source_revision':SOURCE_REV,
    'factory_revision':FACTORY_REV,'base_model':BASE_MODEL,'base_revision':MODEL_REV,
    'gpu':GPU,'max_training_tokens':max_length,'cutoff':CUTOFF,
    'dataset_sha256':audit['source_sha256'],'recommended_config':CONFIG}
(ART/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
(ART/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
print('Model path:',BASE_PATH,'\\nOutput directory:',ADAPTER_DIR,'\\nCutoff:',CUTOFF)
print((ART/'recommended_training.yaml').read_text())
''', 'preflight')
md('''## 3. Fine-tune through LLaMA Board
Use the embedded LLaMA Board below. Select Qwen3-1.7B-Base, and set **Model path** to
the pinned local model path printed above. Select **qwen3_nothink**, dataset
**support_tickets**, and **LoRA**. Use the configuration printed above:
rank 8, alpha 16, all linear targets, learning rate 0.0001, epochs 3,
batch 1, accumulation 8, fp16, seed 42 and the displayed cutoff/output directory.
Preview and save the actual training arguments before starting. Wait for completion.
The server runs in a subprocess so the notebook can continue after training.
''')
code('''
env = os.environ.copy(); env['GRADIO_SHARE']='0'; env['GRADIO_SERVER_PORT']='7860'
ui_log = open(ART/'llamaboard.log','a',buffering=1)
ui_process = subprocess.Popen(['llamafactory-cli','webui'],cwd=FACTORY,env=env,stdout=ui_log,stderr=subprocess.STDOUT)
print('LLaMA Board PID:',ui_process.pid)
''', 'board')
code("print((ART/'llamaboard.log').read_text()[-12000:])\nfrom google.colab import output\noutput.serve_kernel_port_as_iframe(7860, height=900)", 'board_url')
md('## 4. Review completed training\nRun only after LLaMA Board reports training completed. If its output path differs, set ADAPTER_DIR to the actual path before continuing.')
code('''
import matplotlib.pyplot as plt
assert (ADAPTER_DIR/'adapter_config.json').exists(), 'Adapter not saved yet'
assert (ADAPTER_DIR/'adapter_model.safetensors').exists(), 'Missing adapter weights'
state_path = ADAPTER_DIR/'trainer_state.json'
assert state_path.exists(), 'Missing completed trainer state'
state = json.loads(state_path.read_text())
assert state['global_step'] >= state['max_steps'] > 0, 'Training is incomplete'
logs = [json.loads(line) for line in (ADAPTER_DIR/'trainer_log.jsonl').read_text().splitlines() if line.strip()]
loss_logs = [x for x in logs if x.get('loss') is not None]
assert loss_logs, 'No training loss values'
actual_path = ADAPTER_DIR/'training_args.yaml'
assert actual_path.exists(), 'Export the actual training arguments from LLaMA Board before evaluation'
actual_config = validate_training_config(yaml.safe_load(actual_path.read_text()),CONFIG)
manifest['actual_training_config'] = actual_config
plt.figure(figsize=(10,4)); plt.plot([x['current_steps'] for x in loss_logs],[x['loss'] for x in loss_logs])
plt.xlabel('Step'); plt.ylabel('Training loss'); plt.title('Actual LoRA training loss'); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(ART/'training_curve.png',dpi=150); plt.show()
print('Completed steps:',state['global_step'],'First/final loss:',loss_logs[0]['loss'],loss_logs[-1]['loss'])
for name in ('trainer_state.json','trainer_log.jsonl','training_args.yaml','all_results.json','adapter_config.json'):
    if (ADAPTER_DIR/name).exists(): shutil.copy2(ADAPTER_DIR/name,ART/name)
# Release the UI process before loading inference models.
ui_process.terminate(); ui_process.wait(timeout=30); ui_log.close()
manifest['status']='training_completed'; (ART/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
''','training_review')
md('## 5. Baseline, merge and smoke test\nEvaluate the untuned model first. The same pinned tokenizer and non-thinking template are used throughout.')
code('''
import gc
import numpy as np
from transformers import AutoModelForCausalLM
from peft import PeftModel
from tqdm.auto import tqdm
torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
model = AutoModelForCausalLM.from_pretrained(BASE_PATH,torch_dtype=torch.float16,device_map='cuda')
model.eval()
GENERATION_OPTIONS = generation_options(tokenizer,template)
model.generation_config.eos_token_id = GENERATION_OPTIONS['eos_token_id']
model.generation_config.pad_token_id = GENERATION_OPTIONS['pad_token_id']
choices = 'ABCDEFG'
choice_ids = [tokenizer.encode(ch,add_special_tokens=False) for ch in choices]
assert all(len(x)==1 for x in choice_ids) and len(set(x[0] for x in choice_ids))==7
choice_ids = [x[0] for x in choice_ids]
base_system = 'Classify the IT support ticket. Respond with ONLY one letter.\\n'+'\\n'.join(f'{c}) {l}' for c,l in zip(choices,LABELS))
def inputs_for(ticket, system=SYSTEM_PROMPT):
    prompt=tokenizer.apply_chat_template([{'role':'system','content':system},
        {'role':'user','content':'Support Ticket: '+ticket}],tokenize=False,add_generation_prompt=True)
    return tokenizer(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
def base_predict(ticket):
    with torch.inference_mode():
        logits=model(**inputs_for(ticket,base_system)).logits[0,-1,choice_ids]
    return LABELS[int(logits.argmax())]
def tuned_predict(ticket):
    inp=inputs_for(ticket)
    with torch.inference_mode():
        out=model.generate(**inp,**GENERATION_OPTIONS)
    raw=tokenizer.decode(out[0,inp['input_ids'].shape[1]:],skip_special_tokens=True).strip()
    return parse_label(raw),raw
validation=pd.read_csv(ART/'data/val_split.csv')
validation['baseline']=[base_predict(t) for t in tqdm(validation.text,desc='Baseline')]
validation.to_csv(ART/'baseline_predictions.csv',index=False)
smoke=[('Please create a new user account for the employee starting Monday.','Active Directory'),
 ('I cannot access the shared network folder: Access Denied.','Fileservice'),
 ('Outlook and Teams will not sync.','O365'),
 ('Please retire the old server and remove it from monitoring.','EOL'),
 ('Adobe Acrobat fails to install with error 1603.','Software'),
 ('The printer has a paper jam and will not print.','Computer-Services'),
 ('I need help contacting the support team about a general request.','Support general')]
model=PeftModel.from_pretrained(model,str(ADAPTER_DIR)); model.eval()
adapter_results=[tuned_predict(t) for t,_ in smoke]
model=model.merge_and_unload(safe_merge=True); model.eval()
merged_results=[tuned_predict(t) for t,_ in smoke]
verify_merge(adapter_results,merged_results,str(ART/'merge_comparison.json'))
adapter_smoke=[x[0] for x in adapter_results]
merged_smoke=[x[0] for x in merged_results]
model.save_pretrained(MERGED_DIR,safe_serialization=True); tokenizer.save_pretrained(MERGED_DIR)
smoke_df=pd.DataFrame({'ticket':[t for t,_ in smoke],'expected':[l for _,l in smoke],
                       'adapter':adapter_smoke,'merged':merged_smoke})
smoke_df['correct']=smoke_df.expected==smoke_df.merged
smoke_df.to_csv(ART/'smoke_tests.csv',index=False)
display(smoke_df)
print('Smoke tests:',int(smoke_df.correct.sum()),'/',len(smoke_df))
''','baseline_merge')
md('## 6. Evaluate and compare\nEvery held-out row is evaluated. Invalid outputs remain errors. The confusion matrix includes an invalid-output column.')
code('''
results=[tuned_predict(t) for t in tqdm(validation.text,desc='Fine-tuned')]
validation['fine_tuned']=[x[0] for x in results]; validation['raw_output']=[x[1] for x in results]
validation.to_csv(ART/'predictions.csv',index=False)
metrics={'baseline':evaluate_predictions(validation.label.tolist(),validation.baseline.tolist()),
         'fine_tuned':evaluate_predictions(validation.label.tolist(),validation.fine_tuned.tolist())}
metrics['accuracy_delta']=metrics['fine_tuned']['accuracy']-metrics['baseline']['accuracy']
(ART/'metrics.json').write_text(json.dumps(metrics,indent=2))
display(pd.DataFrame(metrics['fine_tuned']['per_class']).T)
print('Baseline accuracy:',metrics['baseline']['accuracy'])
print('Fine-tuned accuracy:',metrics['fine_tuned']['accuracy'])
print('Accuracy delta:',metrics['accuracy_delta'])
import seaborn as sns
cm=np.array(metrics['fine_tuned']['confusion_counts'])[:7,:]
norm=np.divide(cm,cm.sum(axis=1,keepdims=True),out=np.zeros_like(cm,dtype=float),where=cm.sum(axis=1,keepdims=True)!=0)
plt.figure(figsize=(11,7)); sns.heatmap(norm,annot=cm,fmt='d',xticklabels=list(LABELS)+['Invalid'],yticklabels=LABELS,cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('True'); plt.xticks(rotation=35,ha='right'); plt.tight_layout()
plt.savefig(ART/'confusion_matrix.png',dpi=150); plt.show()
x=np.arange(8); fig,ax=plt.subplots(figsize=(12,5))
for offset,key,title in ((-.18,'baseline','Base model'),(.18,'fine_tuned','Fine-tuned')):
    vals=[metrics[key]['per_class'][l]['f1-score'] for l in LABELS]+[metrics[key]['accuracy']]
    ax.bar(x+offset,vals,.36,label=title)
ax.set_xticks(x,list(LABELS)+['Accuracy'],rotation=30,ha='right'); ax.set_ylim(0,1); ax.legend()
fig.tight_layout(); fig.savefig(ART/'baseline_vs_finetuned.png',dpi=150); plt.show()
manifest.update(status='evaluated',heldout_rows=len(validation),smoke_passed=int(smoke_df.correct.sum()),
    smoke_total=len(smoke_df),merge_smoke_agreement=True,metrics=metrics,
    merged_checkpoint=str(MERGED_DIR),adapter_checkpoint=str(ADAPTER_DIR))
(ART/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
print('EVALUATION COMPLETE — preserve artifacts and the executed notebook before disconnecting.')
''','evaluate')
md('''## Save evidence and verified checkpoints
Mount Google Drive when prompted. The cell copies and verifies the evidence, adapter,
tokenizer and merged checkpoint into a persistent Week5-Support-Ticket-Router folder
in My Drive. Move that folder into the designated Week 5 Project folder afterward.
The multi-GB merged checkpoint is retained outside Git. Save the executed notebook too.
''')
code('''
from google.colab import files, drive
drive.mount('/content/drive')
PERSIST = Path('/content/drive/MyDrive/Week5-Support-Ticket-Router')
PERSIST.mkdir(parents=True,exist_ok=True)
preserved = {}
for name,source in [('adapter',ADAPTER_DIR),('merged_model',MERGED_DIR),('training_tokenizer',TOKENIZER_DIR)]:
    preserved[name] = preserve_directory(str(source),str(PERSIST/name))
manifest.update(status='preserved',persistent_directory=str(PERSIST),checkpoint_files=preserved)
(ART/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
preserved['evidence'] = preserve_directory(str(ART),str(PERSIST/'artifacts'))
(PERSIST/'preservation_manifest.json').write_text(json.dumps(preserved,indent=2))
archive=shutil.make_archive(str(ROOT/'week5-evidence'),'zip',ART)
files.download(archive)
print('VERIFIED PERSISTENT CHECKPOINTS AND EVIDENCE:',PERSIST)
''','save')
notebook=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'colab':{'name':'Week5_Support_Ticket_Router.ipynb'},'accelerator':'GPU'})
nbf.write(notebook,ROOT/'notebooks/Week5_Support_Ticket_Router.ipynb')
print('Built',len(cells),'cells')
