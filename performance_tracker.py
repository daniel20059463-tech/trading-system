"""盤前留底績效：缺失不當零；盤後重算不當實盤。"""
from collections import defaultdict
from pathlib import Path
from prediction_audit import collect, summarize, update
import config as cfg


def _daily():
    groups = defaultdict(list)
    for row in collect()[0]:
        if row['eligible'] and row['kind'] == 'point':
            groups[row['date']].append(row)
    rows = []
    for day, records in sorted(groups.items()):
        metrics = summarize(records)
        accuracy = metrics['direction_correct']
        if accuracy['n']:
            rows.append(dict(date=day, acc=accuracy['mean'] * 100, n=accuracy['n'],
                             abs_error_pp=metrics['abs_error_pp']['mean']))
    return rows


def render():
    update()
    report = Path(__file__).parent / 'reports' / 'prediction_audit.md'
    destination = Path(cfg.WIKI_DIR) / '績效追蹤.md'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.read_text(encoding='utf-8'), encoding='utf-8')
    rows = _daily()
    if not rows:
        print('尚無通過時間驗證的盤前留底；已更新缺失與排除報告')
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([r['date'][5:] for r in rows], [r['acc'] for r in rows])
    avg = sum(r['acc'] * r['n'] for r in rows) / sum(r['n'] for r in rows)
    ax.axhline(avg, ls='--', label=f'Sample-weighted accuracy {avg:.1f}%')
    ax.set(ylabel='Direction accuracy (%)', ylim=(0, 100))
    ax.tick_params(axis='x', rotation=60)
    ax.legend()
    fig.tight_layout()
    path = destination.parent / 'charts' / 'performance.png'
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    render()
