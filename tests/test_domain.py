"""核对领域原语的业务数值。"""
import unittest
from analysis_data.io import load
from analysis_data.domain import summarize
class DomainTests(unittest.TestCase):
    def test_raw_counts(self):
        rows=summarize(load('data/normal.json')[0]);r={x['item']:x for x in rows}
        self.assertEqual(r['x1']['counts'],{'normal':2,'blocked':1});self.assertEqual(r['x1']['anchor'],'normal')
        self.assertEqual(rows,summarize(load('data/duplicate-identical.json')[0]));self.assertIsNone(r['x5']['anchor'])
