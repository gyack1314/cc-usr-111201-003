"""汇总稀疏复核票数与外部确认信息。"""
from collections import Counter,defaultdict
def summarize(value):
    labels=value['labels']
    if len(labels)<2 or len(set(labels))!=len(labels):raise ValueError('类别集合错误')
    seen={};votes=defaultdict(Counter)
    for row in value['observations']:
        key=(row['item'],row['reviewer']);label=row['label']
        if not all(key) or label not in labels:raise ValueError('标签记录错误')
        if key in seen and seen[key]!=label:raise ValueError('同一复核对标签冲突')
        if key not in seen:votes[key[0]][label]+=1
        seen[key]=label
    for item,label in value['anchors'].items():
        if item not in votes or label not in labels:raise ValueError('确认项非法')
    return [{'item':item,'counts':dict(sorted(counts.items())),'anchor':value['anchors'].get(item)} for item,counts in sorted(votes.items())]
