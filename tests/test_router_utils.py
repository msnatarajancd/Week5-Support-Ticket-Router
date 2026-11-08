import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import router_utils as r

LABELS = ['Support general', 'Fileservice', 'O365', 'EOL', 'Software', 'Active Directory', 'Computer-Services']

class DataTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([{'text': f'{label} ticket {i}', 'category_truth': label}
                             for label in LABELS for i in range(10)])

    def prepare(self, df):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        df.to_csv(root / 'input.csv', index=False)
        audit = r.prepare_data(str(root / 'input.csv'), str(root / 'out'))
        return root / 'out', audit

    def test_split_is_reproducible_and_disjoint(self):
        root, audit = self.prepare(self.fixture())
        train = json.loads((root / 'train.json').read_text())
        val = pd.read_csv(root / 'val_split.csv')
        self.assertEqual((len(train), len(val)), (56, 14))
        self.assertEqual(set(val.label), set(LABELS))
        texts = {x['messages'][1]['content'].removeprefix('Support Ticket: ') for x in train}
        self.assertFalse(texts & set(val.text))
        self.assertEqual([m['role'] for m in train[0]['messages']], ['system','user','assistant'])
        before = (root / 'val_split.csv').read_bytes()
        r.prepare_data(str(root.parent / 'input.csv'), str(root))
        self.assertEqual(before, (root / 'val_split.csv').read_bytes())

    def test_duplicates_conflicts_and_bad_rows_are_audited(self):
        df = self.fixture()
        extras = pd.DataFrame([
            {'text':'  SUPPORT   GENERAL ticket 0  ', 'category_truth':'Support general'},
            {'text':'conflict', 'category_truth':'O365'},
            {'text':'CONFLICT', 'category_truth':'Software'},
            {'text':'  ', 'category_truth':'EOL'},
            {'text':'unknown', 'category_truth':'other'}])
        _, a = self.prepare(pd.concat([df, extras], ignore_index=True))
        self.assertEqual(a['duplicate_rows_removed'], 1)
        self.assertEqual(a['conflicting_rows_removed'], 2)
        self.assertEqual(a['empty_rows_removed'], 1)
        self.assertEqual(a['unknown_label_rows_removed'], 1)
        self.assertEqual(a['clean_rows'], 70)

    def test_missing_columns_fail(self):
        with self.assertRaises(ValueError): self.prepare(pd.DataFrame({'text':['abc']}))

    def test_missing_and_sparse_classes_fail(self):
        for df in (self.fixture().query("category_truth != 'EOL'"), self.fixture().groupby('category_truth').head(1)):
            with self.assertRaises(ValueError): self.prepare(df)

class MetricTests(unittest.TestCase):
    def test_class_names_do_not_follow_alphabetical_order(self):
        m = r.evaluate_predictions(['O365','O365','Active Directory'], ['O365','Software','Active Directory'])
        self.assertEqual(m['per_class']['Active Directory']['recall'], 1)
        self.assertEqual(m['per_class']['O365']['recall'], .5)
        self.assertAlmostEqual(m['accuracy'], 2/3)

    def test_invalid_is_error_and_visible_in_matrix(self):
        m = r.evaluate_predictions(['Support general','O365'], ['__invalid__','O365'])
        self.assertEqual(m['accuracy'], .5)
        self.assertEqual(m['invalid_count'], 1)
        self.assertEqual(m['confusion_counts'][0][-1], 1)
        self.assertEqual(m['per_class']['Support general']['recall'], 0)
        self.assertAlmostEqual(m['macro_f1'], 1/7)

    def test_label_parser_rejects_explanations(self):
        for label in LABELS: self.assertEqual(r.parse_label('  '+label+'\n'), label)
        for text in ('O365 because email', '', 'AD', 'Support general blah'):
            self.assertEqual(r.parse_label(text), '__invalid__')

    def test_invalid_metric_inputs_fail(self):
        for truth,pred in [([],[]), (['O365'],[]), (['bad'],['O365'])]:
            with self.assertRaises(ValueError): r.evaluate_predictions(truth,pred)

if __name__ == '__main__': unittest.main()

class RuntimeSafetyTests(unittest.TestCase):
    def test_frozen_base_model_rejects_replaced_eos(self):
        r.validate_native_eos(151643,151643)
        with self.assertRaisesRegex(ValueError,'native EOS'):
            r.validate_native_eos(151643,151645)

    def test_generation_stops_at_training_template_eos(self):
        class Tokenizer: eos_token_id = 151645
        class Template:
            def get_stop_token_ids(self, tokenizer): return [151645, 151643]
        options = r.generation_options(Tokenizer(), Template())
        self.assertEqual(options['eos_token_id'], [151645,151643])
        self.assertEqual(options['pad_token_id'],151645)

    def test_raw_merge_disagreement_is_saved_even_when_labels_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'comparison.json'
            with self.assertRaises(ValueError):
                r.verify_merge([('bad output A','__invalid__')],[('bad output B','__invalid__')],str(path))
            self.assertTrue(path.exists())
            self.assertFalse(json.loads(path.read_text())['matches'])

    def test_actual_training_configuration_must_match_critical_settings(self):
        expected={'model_name_or_path':'model','template':'qwen3_nothink','dataset':'support_tickets','cutoff_len':1024,'finetuning_type':'lora'}
        self.assertEqual(r.validate_training_config(dict(expected),expected), expected)
        for key in expected:
            actual=dict(expected); actual[key]='WRONG'
            with self.assertRaises(ValueError): r.validate_training_config(actual,expected)
        with self.assertRaises(ValueError): r.validate_training_config({},expected)

    def test_preserved_checkpoint_is_verified_by_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); src=root/'source'; src.mkdir(); (src/'model.bin').write_bytes(b'checkpoint')
            manifest=r.preserve_directory(str(src),str(root/'persistent'))
            self.assertEqual((root/'persistent/model.bin').read_bytes(),b'checkpoint')
            self.assertEqual(manifest[0]['bytes'],10)
            self.assertEqual(len(manifest[0]['sha256']),64)
