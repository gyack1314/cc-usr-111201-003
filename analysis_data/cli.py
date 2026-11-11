"""展示已有的资料基本统计与Dawid-Skene潜在类别分析，不改变原文件。"""
import argparse
import json
from .io import load
from .domain import summarize
from . import ds


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('inspect', help='原始资料摘要与基础统计').add_argument('file')
    sub.add_parser('fit', help='多初值Dawid-Skene MAP-EM拟合与导出').add_argument('file')
    objective_parser = sub.add_parser('objective', help='用给定参数独立重算目标对数后验')
    objective_parser.add_argument('file')
    objective_parser.add_argument('parameters')
    args = parser.parse_args()
    value, digest = load(args.file)
    if args.command == 'inspect':
        _print({'source_sha256': digest, 'summary': summarize(value)})
    elif args.command == 'fit':
        report = ds.fit(value)
        report['source_sha256'] = digest
        _print(report)
    else:
        parameters, parameters_digest = load(args.parameters)
        _print({'source_sha256': digest, 'parameters_sha256': parameters_digest,
                'objective': ds.parameters_objective(value, parameters)})


if __name__ == '__main__':
    main()
