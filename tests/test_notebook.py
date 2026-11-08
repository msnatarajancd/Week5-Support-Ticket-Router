import unittest
from pathlib import Path
import nbformat

class NotebookTests(unittest.TestCase):
    def test_notebook_is_valid_and_code_compiles(self):
        path = Path('notebooks/Week5_Support_Ticket_Router.ipynb')
        self.assertTrue(path.exists(), 'Deliverable notebook must exist')
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'cell_{i}', 'exec')
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)

    def test_embedded_helper_runs_real_data_preparation(self):
        import tempfile
        import pandas as pd
        path = Path('notebooks/Week5_Support_Ticket_Router.ipynb')
        self.assertTrue(path.exists())
        nb = nbformat.read(path, as_version=4)
        cell = next(c for c in nb.cells if c.metadata.get('name') == 'helper')
        namespace = {}
        exec(cell.source, namespace)
        with tempfile.TemporaryDirectory() as tmp:
            audit = namespace['prepare_data']('data/support_tickets.csv', tmp)
            self.assertGreater(audit['train_rows'], 0)
            self.assertEqual(audit['normalized_overlap'], 0)
            self.assertEqual(set(pd.read_csv(Path(tmp)/'val_split.csv').label), set(namespace['LABELS']))

if __name__ == '__main__': unittest.main()
