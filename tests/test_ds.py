"""核对 Dawid-Skene 潜在类别推断的数值、停止规则与导出契约。"""
import contextlib
import copy
import io as text_io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analysis_data import cli, ds
from analysis_data.io import load


def normal_value():
    return load('data/normal.json')[0]


def tiny_value(**overrides):
    value = {'labels': ['a', 'b'],
             'observations': [{'item': 'i1', 'reviewer': 'r1', 'label': 'a'},
                              {'item': 'i1', 'reviewer': 'r2', 'label': 'b'},
                              {'item': 'i2', 'reviewer': 'r1', 'label': 'a'},
                              {'item': 'i2', 'reviewer': 'r2', 'label': 'a'},
                              {'item': 'i3', 'reviewer': 'r1', 'label': 'b'},
                              {'item': 'i3', 'reviewer': 'r2', 'label': 'b'}],
             'anchors': {'i2': 'a', 'i3': 'b'},
             'prior_alpha': [2, 2],
             'confusion_alpha': [[4, 2], [2, 4]],
             'starts': [{'id': 's', 'diagonal': 0.8}],
             'max_iterations': 300,
             'parameter_tolerance': 1e-10,
             'objective_tolerance': 1e-12,
             'decrease_tolerance': 1e-07}
    value.update(overrides)
    return value


def run_cli(argv):
    out = text_io.StringIO()
    with mock.patch('sys.argv', argv), contextlib.redirect_stdout(out):
        cli.main()
    return json.loads(out.getvalue())


class PrepareTests(unittest.TestCase):
    def test_conflict_rejected(self):
        with self.assertRaises(ValueError):
            ds.fit(load('data/invalid-conflict.json')[0])

    def test_duplicate_identical_merged(self):
        base = ds.fit(normal_value())
        merged = ds.fit(load('data/duplicate-identical.json')[0])
        self.assertAlmostEqual(base['starts'][base['best_start']]['final']['objective'],
                               merged['starts'][merged['best_start']]['final']['objective'],
                               places=12)

    def test_dirichlet_alpha_must_exceed_one(self):
        with self.assertRaises(ValueError):
            ds.prepare(tiny_value(prior_alpha=[1, 2]))
        with self.assertRaises(ValueError):
            ds.prepare(tiny_value(confusion_alpha=[[4, 2], [2, 1]]))

    def test_explicit_start_validation(self):
        explicit = {'id': 'e', 'prior': [0.5, 0.5], 'confusion': [[0.9, 0.1], [0.2, 0.8]]}
        report = ds.fit(tiny_value(starts=[explicit]))
        self.assertEqual(report['starts'][0]['status'], 'converged')
        with self.assertRaises(ValueError):
            ds.prepare(tiny_value(starts=[{'id': 'e', 'prior': [0.5, 0.4],
                                           'confusion': [[0.9, 0.1], [0.2, 0.8]]}]))
        with self.assertRaises(ValueError):
            ds.prepare(tiny_value(starts=[{'id': 'e', 'prior': [0.5, 0.5],
                                           'confusion': [[0.9, 0.1], [0.0, 1.0]]}]))
        with self.assertRaises(ValueError):
            ds.prepare(tiny_value(starts=[{'id': 'e', 'diagonal': 0.8, 'prior': [0.5, 0.5],
                                           'confusion': [[0.9, 0.1], [0.2, 0.8]]}]))

    def test_missing_decrease_tolerance_rejected(self):
        value = tiny_value()
        del value['decrease_tolerance']
        with self.assertRaises(KeyError):
            ds.prepare(value)


class FitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = ds.fit(normal_value())

    def test_objective_monotone_and_converged(self):
        for start in self.report['starts']:
            self.assertEqual(start['status'], 'converged')
            self.assertTrue(start['converged'])
            objectives = [h['objective'] for h in start['iterations']]
            for previous, current in zip(objectives, objectives[1:]):
                self.assertGreaterEqual(current, previous - 1e-9)
            self.assertLess(start['iterations'][-1]['max_change'], 1e-08)

    def test_iteration_limit_is_not_convergence(self):
        report = ds.fit(tiny_value(max_iterations=1))
        start = report['starts'][0]
        self.assertEqual(start['status'], 'iteration_limit')
        self.assertFalse(start['converged'])
        self.assertIn('上限', start['stop_reason'])

    def test_objective_decrease_stops_and_explains(self):
        model = ds.prepare(tiny_value())
        objectives = iter([0.0, -1.0])
        result = ds._run_start(model, model.start_specs[0],
                               objective=lambda *args: next(objectives))
        self.assertEqual(result['status'], 'objective_decrease')
        self.assertFalse(result['converged'])
        self.assertIn('下降', result['stop_reason'])
        self.assertEqual(result['final']['objective'], 0.0)

    def test_tie_breaks_by_start_index(self):
        starts = [{'id': 'first', 'diagonal': 0.8}, {'id': 'second', 'diagonal': 0.8}]
        report = ds.fit(tiny_value(starts=starts))
        self.assertEqual(len(report['starts']), 2)
        self.assertEqual(report['starts'][0]['final']['objective'],
                         report['starts'][1]['final']['objective'])
        self.assertEqual(report['best_start'], 0)

    def test_anchor_locked_one_hot(self):
        items = {x['item']: x for x in self.report['items']}
        self.assertEqual(items['x1']['posterior'], {'normal': 1.0, 'contact': 0.0, 'blocked': 0.0})
        self.assertEqual(items['x1']['entropy'], 0.0)
        self.assertEqual(items['x1']['anchor'], 'normal')
        contrary = tiny_value(anchors={'i1': 'a', 'i2': 'a', 'i3': 'b'})
        locked = {x['item']: x for x in ds.fit(contrary)['items']}
        self.assertEqual(locked['i1']['posterior'], {'a': 1.0, 'b': 0.0})
        self.assertEqual(locked['i1']['entropy'], 0.0)

    def test_sparse_reviewer_rows_normalized(self):
        reviewers = {r['reviewer']: r for r in self.report['reviewers']}
        sparse = reviewers['r4']
        self.assertEqual(sparse['labels_given'], 1)
        for row in sparse['confusion']:
            self.assertAlmostEqual(sum(row), 1.0, places=12)
            self.assertTrue(all(math.isfinite(v) and v > 0 for v in row))
        self.assertAlmostEqual(sum(sparse['row_effective_counts']), 1.0, places=12)

    def test_unanchored_isolated_component_visible(self):
        components = {tuple(c['items']): c for c in self.report['components']}
        self.assertEqual(len(components), 2)
        isolated = components[('x5',)]
        self.assertEqual(isolated['reviewers'], ['r4'])
        self.assertFalse(isolated['anchored'])
        self.assertIn('类别方向', isolated['warning'])
        anchored = components[('x1', 'x2', 'x3', 'x4')]
        self.assertTrue(anchored['anchored'])
        self.assertEqual(anchored['anchor_items'], ['x1', 'x3'])
        items = {x['item']: x for x in self.report['items']}
        self.assertFalse(items['x5']['component_anchored'])
        self.assertAlmostEqual(sum(items['x5']['posterior'].values()), 1.0, places=12)

    def test_unanchored_material_still_fits(self):
        report = ds.fit(load('data/unanchored.json')[0])
        self.assertTrue(all(not c['anchored'] for c in report['components']))
        self.assertTrue(all(not x['component_anchored'] for x in report['items']))
        self.assertEqual(report['starts'][report['best_start']]['status'], 'converged')

    def test_strong_disagreement_splits_posterior(self):
        items = {x['item']: x for x in ds.fit(tiny_value())['items']}
        split = items['i1']
        self.assertGreater(split['posterior']['a'], 0.3)
        self.assertLess(split['posterior']['a'], 0.7)
        self.assertGreater(split['entropy'], 0.5)
        self.assertLess(split['entropy'], math.log(2) + 1e-9)

    def test_label_permutation_equivariant(self):
        order = [2, 0, 1]
        value = normal_value()
        permuted = copy.deepcopy(value)
        permuted['labels'] = [value['labels'][i] for i in order]
        permuted['prior_alpha'] = [value['prior_alpha'][i] for i in order]
        permuted['confusion_alpha'] = [[value['confusion_alpha'][order[z]][order[l]]
                                        for l in range(3)] for z in range(3)]
        base, moved = self.report, ds.fit(permuted)
        self.assertAlmostEqual(base['starts'][base['best_start']]['final']['objective'],
                               moved['starts'][moved['best_start']]['final']['objective'],
                               delta=1e-7)
        base_items = {x['item']: x for x in base['items']}
        for item in moved['items']:
            for label, probability in item['posterior'].items():
                self.assertAlmostEqual(probability, base_items[item['item']]['posterior'][label],
                                       delta=1e-7)
        base_reviewers = {r['reviewer']: r for r in base['reviewers']}
        for reviewer in moved['reviewers']:
            expected = base_reviewers[reviewer['reviewer']]['confusion']
            for z in range(3):
                for l in range(3):
                    self.assertAlmostEqual(reviewer['confusion'][z][l],
                                           expected[order[z]][order[l]], delta=1e-7)

    def test_objective_recomputed_from_exported_parameters(self):
        value = normal_value()
        best = self.report['starts'][self.report['best_start']]
        model = ds.prepare(value)
        confusion = [best['final']['confusion'][name] for name in model.reviewers]
        objective = ds.log_posterior(model, best['final']['prior'], confusion)
        self.assertAlmostEqual(objective, best['final']['objective'], places=12)
        self.assertAlmostEqual(ds.parameters_objective(value, best['final']),
                               best['final']['objective'], places=12)

    def test_parameters_objective_rejects_mismatch(self):
        best = self.report['starts'][self.report['best_start']]
        broken = {'prior': best['final']['prior'],
                  'confusion': {'outsider': best['final']['confusion']['r1']}}
        with self.assertRaises(ValueError):
            ds.parameters_objective(normal_value(), broken)

    def test_export_contract(self):
        dumped = json.dumps(self.report, ensure_ascii=False, allow_nan=False)
        self.assertNotIn('"accuracy"', dumped)
        self.assertEqual(self.report['tolerances']['decrease'], 1e-07)
        for item in self.report['items']:
            self.assertEqual(set(item['posterior']), set(self.report['labels']))
            self.assertTrue(item['votes'])
        for start in self.report['starts']:
            self.assertTrue(start['stop_reason'])
            self.assertIsNone(start['iterations'][0]['max_change'])


class CliTests(unittest.TestCase):
    def test_fit_then_objective_round_trip(self):
        report = run_cli(['cli', 'fit', 'data/normal.json'])
        self.assertEqual(len(report['source_sha256']), 64)
        best = report['starts'][report['best_start']]
        with tempfile.TemporaryDirectory() as directory:
            parameters = Path(directory) / '参数.json'
            parameters.write_text(json.dumps(best['final']), encoding='utf8')
            result = run_cli(['cli', 'objective', 'data/normal.json', str(parameters)])
        self.assertEqual(result['source_sha256'], report['source_sha256'])
        self.assertAlmostEqual(result['objective'], best['final']['objective'], places=12)

    def test_inspect_unchanged(self):
        result = run_cli(['cli', 'inspect', 'data/normal.json'])
        self.assertIn('summary', result)
        self.assertEqual(len(result['source_sha256']), 64)


if __name__ == '__main__':
    unittest.main()
