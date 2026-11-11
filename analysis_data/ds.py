"""多类别 Dawid-Skene 潜在类别 MAP 推断。

数学口径见 docs/data-contract.md：目标为带 Dirichlet 先验的对数后验
（先验归一常数省略，全部初值与独立核对共用同一省略），混淆矩阵行是
真类别、列是观察标签。anchor 后验为严格 one-hot 并在推断中固定；缺席
记录不补票。潜在类别概率是模型推断而非外部真值，不得回灌计算复核员
准确率。
"""
import math
from .io import finite, unique

_NORMALIZE_TOLERANCE = 1e-9


class Model:
    """校验后的推断问题：索引化投票、anchor、Dirichlet先验与停止配置。"""

    def __init__(self, labels, items, reviewers, item_votes, reviewer_votes,
                 anchor_of, votes_by_item, prior_alpha, confusion_alpha,
                 start_specs, max_iterations, parameter_tolerance,
                 objective_tolerance, decrease_tolerance):
        self.labels = labels
        self.K = len(labels)
        self.items = items
        self.reviewers = reviewers
        self.item_votes = item_votes          # 每item: tuple((reviewer索引, 标签索引), ...)
        self.reviewer_votes = reviewer_votes  # 每reviewer: tuple((item索引, 标签索引), ...)
        self.anchor_of = anchor_of            # item索引 -> 标签索引
        self.votes_by_item = votes_by_item    # 每item: {reviewer名: 标签}
        self.prior_alpha = prior_alpha
        self.confusion_alpha = confusion_alpha
        self.start_specs = start_specs
        self.max_iterations = max_iterations
        self.parameter_tolerance = parameter_tolerance
        self.objective_tolerance = objective_tolerance
        self.decrease_tolerance = decrease_tolerance


def _num(value):
    return float(finite(value))


def _check_distribution(row, what):
    values = [_num(x) for x in row]
    if any(v <= 0 for v in values):
        raise ValueError(f'{what}必须严格为正')
    if abs(sum(values) - 1.0) > _NORMALIZE_TOLERANCE:
        raise ValueError(f'{what}必须归一')
    return values


def prepare(value):
    """校验资料与配置，返回索引化的Model；重复同标签合并、冲突标签拒绝。"""
    if not isinstance(value, dict):
        raise ValueError('资料顶层必须是对象')
    labels = value['labels']
    if (not isinstance(labels, list) or len(labels) < 2
            or any(not isinstance(x, str) or not x for x in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError('类别集合错误')
    K = len(labels)
    label_index = {name: z for z, name in enumerate(labels)}
    seen = {}
    for row in value['observations']:
        item, reviewer, label = row['item'], row['reviewer'], row['label']
        if not (isinstance(item, str) and item and isinstance(reviewer, str) and reviewer):
            raise ValueError('标签记录错误')
        if label not in label_index:
            raise ValueError('标签记录错误')
        key = (item, reviewer)
        if key in seen and seen[key] != label:
            raise ValueError('同一复核对标签冲突')
        seen[key] = label
    items = sorted({key[0] for key in seen})
    reviewers = sorted({key[1] for key in seen})
    item_index = {name: i for i, name in enumerate(items)}
    reviewer_index = {name: r for r, name in enumerate(reviewers)}
    item_votes = [[] for _ in items]
    for (item, reviewer), label in seen.items():
        item_votes[item_index[item]].append((reviewer_index[reviewer], label_index[label]))
    item_votes = [tuple(sorted(votes)) for votes in item_votes]
    reviewer_votes = [[] for _ in reviewers]
    for i, votes in enumerate(item_votes):
        for r, l in votes:
            reviewer_votes[r].append((i, l))
    reviewer_votes = [tuple(sorted(votes)) for votes in reviewer_votes]
    anchors = value['anchors']
    if not isinstance(anchors, dict):
        raise ValueError('确认项非法')
    anchor_of = {}
    for item, label in anchors.items():
        if item not in item_index or label not in label_index:
            raise ValueError('确认项非法')
        anchor_of[item_index[item]] = label_index[label]
    prior_alpha = value['prior_alpha']
    if not isinstance(prior_alpha, list) or len(prior_alpha) != K:
        raise ValueError('先验Dirichlet参数维度错误')
    prior_alpha = [_num(x) for x in prior_alpha]
    if any(a <= 1 for a in prior_alpha):
        raise ValueError('先验Dirichlet参数必须大于1')
    confusion_alpha = value['confusion_alpha']
    if (not isinstance(confusion_alpha, list) or len(confusion_alpha) != K
            or any(not isinstance(row, list) or len(row) != K for row in confusion_alpha)):
        raise ValueError('混淆Dirichlet参数维度错误')
    confusion_alpha = [[_num(x) for x in row] for row in confusion_alpha]
    if any(a <= 1 for row in confusion_alpha for a in row):
        raise ValueError('混淆Dirichlet参数必须大于1')
    start_specs = value['starts']
    if not isinstance(start_specs, list) or not start_specs:
        raise ValueError('至少需要一个初值')
    unique(start_specs, 'id')
    specs = []
    for spec in start_specs:
        has_diagonal = 'diagonal' in spec
        has_explicit = 'prior' in spec or 'confusion' in spec
        if has_diagonal == has_explicit:
            raise ValueError('初值必须且只能给对角值或完整显式参数')
        if has_diagonal:
            diagonal = _num(spec['diagonal'])
            if not 0 < diagonal < 1:
                raise ValueError('对角值必须在(0,1)开区间内')
            specs.append({'id': spec['id'], 'diagonal': diagonal})
        else:
            if 'prior' not in spec or 'confusion' not in spec:
                raise ValueError('显式初值必须同时给prior与confusion')
            prior, confusion = spec['prior'], spec['confusion']
            if not isinstance(prior, list) or len(prior) != K:
                raise ValueError('显式初值维度错误')
            if (not isinstance(confusion, list) or len(confusion) != K
                    or any(not isinstance(row, list) or len(row) != K for row in confusion)):
                raise ValueError('显式初值维度错误')
            specs.append({'id': spec['id'],
                          'prior': _check_distribution(prior, '显式初值先验'),
                          'confusion': [_check_distribution(row, '显式初值混淆行') for row in confusion]})
    max_iterations = value['max_iterations']
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError('迭代上限必须是正整数')
    parameter_tolerance = _num(value['parameter_tolerance'])
    objective_tolerance = _num(value['objective_tolerance'])
    decrease_tolerance = _num(value['decrease_tolerance'])
    if min(parameter_tolerance, objective_tolerance, decrease_tolerance) <= 0:
        raise ValueError('容差必须为正')
    votes_by_item = [{reviewers[r]: labels[l] for r, l in votes} for votes in item_votes]
    return Model(labels, items, reviewers, item_votes, reviewer_votes,
                 anchor_of, votes_by_item, prior_alpha, confusion_alpha, specs,
                 max_iterations, parameter_tolerance, objective_tolerance,
                 decrease_tolerance)


def _logsumexp(values):
    peak = max(values)
    return peak + math.log(sum(math.exp(v - peak) for v in values))


def log_posterior(model, prior, confusion):
    """独立重算目标：带Dirichlet先验的对数后验（省略先验归一常数）。

    prior为长度K的列表，confusion按reviewer索引给出KxK行（行=真类别）。
    """
    K = model.K
    total = 0.0
    for i, votes in enumerate(model.item_votes):
        anchor = model.anchor_of.get(i)
        if anchor is None:
            total += _logsumexp([math.log(prior[z]) + sum(math.log(confusion[r][z][l]) for r, l in votes)
                                 for z in range(K)])
        else:
            total += math.log(prior[anchor]) + sum(math.log(confusion[r][anchor][l]) for r, l in votes)
    total += sum((a - 1.0) * math.log(p) for a, p in zip(model.prior_alpha, prior))
    for rows in confusion:
        for z in range(K):
            for l in range(K):
                total += (model.confusion_alpha[z][l] - 1.0) * math.log(rows[z][l])
    return total


def _posteriors(model, prior, confusion):
    """E步：log-sum-exp稳定归一；anchor为严格one-hot。"""
    K = model.K
    result = []
    for i, votes in enumerate(model.item_votes):
        anchor = model.anchor_of.get(i)
        if anchor is not None:
            result.append([1.0 if z == anchor else 0.0 for z in range(K)])
            continue
        logs = [math.log(prior[z]) + sum(math.log(confusion[r][z][l]) for r, l in votes)
                for z in range(K)]
        peak = max(logs)
        weights = [math.exp(v - peak) for v in logs]
        total = sum(weights)
        result.append([w / total for w in weights])
    return result


def _maximization(model, q):
    """M步：pi含全部item（anchor计one-hot），混淆行只累计实际所给标签，伪计数alpha-1。"""
    K = model.K
    counts = [0.0] * K
    for qi in q:
        for z in range(K):
            counts[z] += qi[z]
    prior = [counts[z] + model.prior_alpha[z] - 1.0 for z in range(K)]
    total = sum(prior)
    prior = [p / total for p in prior]
    confusion = []
    for r in range(len(model.reviewers)):
        rows = [[model.confusion_alpha[z][l] - 1.0 for l in range(K)] for z in range(K)]
        for i, l in model.reviewer_votes[r]:
            qi = q[i]
            for z in range(K):
                rows[z][l] += qi[z]
        for z in range(K):
            total = sum(rows[z])
            rows[z] = [v / total for v in rows[z]]
        confusion.append(rows)
    return prior, confusion


def _start_params(model, spec):
    K = model.K
    if 'diagonal' in spec:
        diagonal = spec['diagonal']
        off = (1.0 - diagonal) / (K - 1)
        prior = [1.0 / K] * K
        confusion = [[[diagonal if l == z else off for l in range(K)] for z in range(K)]
                     for _ in model.reviewers]
    else:
        prior = list(spec['prior'])
        confusion = [[list(row) for row in spec['confusion']] for _ in model.reviewers]
    return prior, confusion


def _snapshot(iteration, objective, max_change, prior, confusion):
    return {'iteration': iteration, 'objective': objective, 'max_change': max_change,
            'prior': list(prior), 'confusion': [[list(row) for row in rows] for rows in confusion]}


def _max_change(old_prior, old_confusion, new_prior, new_confusion):
    changes = [abs(a - b) for a, b in zip(old_prior, new_prior)]
    for old_rows, new_rows in zip(old_confusion, new_confusion):
        for old_row, new_row in zip(old_rows, new_rows):
            changes.extend(abs(a - b) for a, b in zip(old_row, new_row))
    return max(changes)


def _run_start(model, spec, objective=log_posterior):
    """单初值MAP-EM；每次更新保存目标、参数与最大变化。目标下降超容差即停并解释。"""
    prior, confusion = _start_params(model, spec)
    history = [_snapshot(0, objective(model, prior, confusion), None, prior, confusion)]
    status = 'iteration_limit'
    converged = False
    reason = f'达到迭代上限{model.max_iterations}，未同时满足参数与目标容差，不得视为收敛'
    for iteration in range(1, model.max_iterations + 1):
        q = _posteriors(model, prior, confusion)
        new_prior, new_confusion = _maximization(model, q)
        new_objective = objective(model, new_prior, new_confusion)
        max_change = _max_change(prior, confusion, new_prior, new_confusion)
        delta = new_objective - history[-1]['objective']
        history.append(_snapshot(iteration, new_objective, max_change, new_prior, new_confusion))
        if delta < -model.decrease_tolerance:
            status = 'objective_decrease'
            reason = (f'第{iteration}次更新目标下降{-delta:.6g}，超出浮点下降容差'
                      f'{model.decrease_tolerance:.6g}；MAP-EM的对数后验目标不应出现数值误差之外的'
                      '下降，已停止并保留历史目标最高的参数')
            break
        prior, confusion = new_prior, new_confusion
        if max_change < model.parameter_tolerance and delta < model.objective_tolerance:
            status = 'converged'
            converged = True
            reason = (f'参数最大变化{max_change:.6g}低于容差{model.parameter_tolerance:.6g}且'
                      f'目标增量{delta:.6g}低于容差{model.objective_tolerance:.6g}，判定收敛')
            break
    best = max(history, key=lambda h: h['objective'])
    return {'status': status, 'converged': converged, 'stop_reason': reason,
            'iterations': history,
            'final': {'objective': best['objective'], 'prior': best['prior'],
                      'confusion': best['confusion']}}


def _components(model):
    """reviewer-item二部图连通分量；无anchor分量类别方向不可靠。"""
    item_comp = [None] * len(model.items)
    reviewer_comp = [None] * len(model.reviewers)
    components = []
    seeds = [('item', i) for i in range(len(model.items))]
    seeds += [('reviewer', r) for r in range(len(model.reviewers))]
    for kind, index in seeds:
        assigned = item_comp[index] if kind == 'item' else reviewer_comp[index]
        if assigned is not None:
            continue
        cid = len(components)
        comp_items, comp_reviewers = [], []
        queue = [(kind, index)]
        while queue:
            node_kind, node = queue.pop()
            if node_kind == 'item':
                if item_comp[node] is not None:
                    continue
                item_comp[node] = cid
                comp_items.append(node)
                queue.extend(('reviewer', r) for r, _ in model.item_votes[node])
            else:
                if reviewer_comp[node] is not None:
                    continue
                reviewer_comp[node] = cid
                comp_reviewers.append(node)
                queue.extend(('item', i) for i, _ in model.reviewer_votes[node])
        comp_items.sort()
        comp_reviewers.sort()
        components.append({'items': comp_items, 'reviewers': comp_reviewers,
                           'anchored': any(i in model.anchor_of for i in comp_items)})
    return components, item_comp, reviewer_comp


def fit(value):
    """多初值MAP-EM拟合并导出报告；各次结果全部保留，按最终目标择优、同分取编号较小者。"""
    model = prepare(value)
    starts = []
    for index, spec in enumerate(model.start_specs):
        result = _run_start(model, spec)
        result['index'] = index
        result['id'] = spec['id']
        starts.append(result)
    best_start = 0
    for index in range(1, len(starts)):
        if starts[index]['final']['objective'] > starts[best_start]['final']['objective']:
            best_start = index
    components, item_comp, reviewer_comp = _components(model)
    final = starts[best_start]['final']
    posterior = _posteriors(model, final['prior'], final['confusion'])
    labels = model.labels
    K = model.K
    components_out = []
    for cid, comp in enumerate(components):
        entry = {'id': cid,
                 'items': [model.items[i] for i in comp['items']],
                 'reviewers': [model.reviewers[r] for r in comp['reviewers']],
                 'anchored': comp['anchored'],
                 'anchor_items': sorted(model.items[i] for i in comp['items'] if i in model.anchor_of)}
        if not comp['anchored']:
            entry['warning'] = '该分量无anchor，类别方向无法可靠确定，潜在类别概率不得当作已确认真值'
        components_out.append(entry)
    items_out = []
    for i, item in enumerate(model.items):
        anchor = model.anchor_of.get(i)
        q = posterior[i]
        entropy = 0.0 if anchor is not None else -sum(p * math.log(p) for p in q if p > 0)
        items_out.append({'item': item,
                          'posterior': {labels[z]: q[z] for z in range(K)},
                          'entropy': entropy,
                          'anchor': None if anchor is None else labels[anchor],
                          'component': item_comp[i],
                          'component_anchored': components[item_comp[i]]['anchored'],
                          'votes': model.votes_by_item[i]})
    reviewers_out = []
    for r, name in enumerate(model.reviewers):
        effective = [0.0] * K
        for i, _l in model.reviewer_votes[r]:
            qi = posterior[i]
            for z in range(K):
                effective[z] += qi[z]
        reviewers_out.append({'reviewer': name,
                              'confusion': final['confusion'][r],
                              'row_effective_counts': effective,
                              'labels_given': len(model.reviewer_votes[r]),
                              'component': reviewer_comp[r]})

    def named(confusion):
        return {model.reviewers[r]: confusion[r] for r in range(len(model.reviewers))}

    starts_out = [{'index': s['index'], 'id': s['id'], 'status': s['status'],
                   'converged': s['converged'], 'stop_reason': s['stop_reason'],
                   'iterations': [{'iteration': h['iteration'], 'objective': h['objective'],
                                   'max_change': h['max_change'], 'prior': h['prior'],
                                   'confusion': named(h['confusion'])} for h in s['iterations']],
                   'final': {'objective': s['final']['objective'], 'prior': s['final']['prior'],
                             'confusion': named(s['final']['confusion'])}}
                  for s in starts]
    return {'labels': list(labels),
            'conventions': '混淆矩阵行是真类别、列是观察标签，行列顺序与labels一致；后验熵为自然对数，anchor熵为零',
            'tolerances': {'parameter': model.parameter_tolerance,
                           'objective': model.objective_tolerance,
                           'decrease': model.decrease_tolerance},
            'max_iterations': model.max_iterations,
            'components': components_out,
            'starts': starts_out,
            'best_start': best_start,
            'items': items_out,
            'reviewers': reviewers_out,
            'notes': ['潜在类别概率与混淆矩阵为MAP推断结果，不是外部真值；禁止用推断多数类别回算复核员准确率',
                      '目标为含Dirichlet先验的对数后验（省略先验归一常数）；观测似然单独下降不等于失败，以报告的对数后验为准']}


def parameters_objective(value, parameters):
    """用外部给定参数独立重算目标（对数后验），供结果核对。

    parameters为{'prior': [...], 'confusion': {reviewer名: KxK行}}，
    可直接使用fit输出中某个final（多余键被忽略）。
    """
    model = prepare(value)
    if not isinstance(parameters, dict):
        raise ValueError('参数顶层必须是对象')
    prior = parameters['prior']
    if not isinstance(prior, list) or len(prior) != model.K:
        raise ValueError('参数先验维度错误')
    prior = _check_distribution(prior, '参数先验')
    confusion = parameters['confusion']
    if not isinstance(confusion, dict) or set(confusion) != set(model.reviewers):
        raise ValueError('参数混淆必须按reviewer给出且集合与资料一致')
    rows = []
    for name in model.reviewers:
        matrix = confusion[name]
        if (not isinstance(matrix, list) or len(matrix) != model.K
                or any(not isinstance(row, list) or len(row) != model.K for row in matrix)):
            raise ValueError('参数混淆维度错误')
        rows.append([_check_distribution(row, '参数混淆行') for row in matrix])
    return log_posterior(model, prior, rows)
