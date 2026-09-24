"""格式化單一最終區間；推播函式才會連線 Discord。"""
from datetime import datetime, timedelta, timezone


def build_message(q_results):
    shadow = any(r.get('evidence_status') != 'approved' for r in q_results)
    lines = ['📊 **下一交易日單一預估區間（研究觀察）**' if shadow else
             '📊 **下一交易日單一預估區間**']
    for r in sorted(q_results, key=lambda r: -r['p_up']):
        target = r.get('target_date', '待交易日確認')
        base = float(r.get('last_close', 0))
        low = r.get('q25_price', base * (1 + r['q25_pct'] / 100))
        high = r.get('q75_price', base * (1 + r['q75_pct'] / 100))
        center = r.get('q50_price', base * (1 + r['q50_pct'] / 100))
        lines.extend([
            f"**{r['name']}**｜資料截至 {r.get('data_as_of', '未知')}｜目標 {target}",
            f"預估收盤區間：{low:.2f}～{high:.2f}",
            f"中心價：{center:.2f}",
        ])
    lines.append('每檔只顯示一組最終價格區間；其他區間僅供內部驗證。')
    return '\n'.join(lines)


def build_context_message(payload):
    lines = ['📊 **今日最終預估區間（已納入昨夜美股）**',
             f"目標日：{payload['target_date']}｜每檔只顯示一組價格區間。"]
    for row in payload['records']:
        center_price = row['last_close'] * (1 + row['center_pct'] / 100)
        lines.extend([f"**{row['name']} {row['ticker']}**",
                      f"預估收盤：{row['q25_price']:.2f}～{row['q75_price']:.2f}｜中心 {center_price:.2f}"])
    lines.append('其他寬區間保留在內部稽核，不再推播。')
    return '\n'.join(lines)


def push_context_to_discord(payload, root=None):
    from pathlib import Path
    from prediction_audit import HERE
    from intraday_monitor import send_discord, WEBHOOK
    root = Path(root or HERE)
    if not WEBHOOK:
        return False
    marker = root / 'live_evidence' / 'us_context_interval' / 'pushed' / f"{payload['target_date']}.json"
    if marker.exists():
        return True
    ok = all(send_discord(part) for part in chunks(build_context_message(payload)))
    if ok:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text('{"status":"sent"}', encoding='utf-8')
    return ok


def chunks(message, limit=1900):
    result, current = [], ''
    for line in message.splitlines():
        if len(current) + len(line) + 1 > limit:
            if current:
                result.append(current)
            current = ''
        while len(line) > limit:
            result.append(line[:limit])
            line = line[limit:]
        current = current + '\n' + line if current else line
    if current:
        result.append(current)
    return result


def push_to_discord(q_results=None):
    if q_results is None:
        from prediction_audit import HERE, read_json
        today = datetime.now(timezone(timedelta(hours=8))).date().isoformat()
        path = HERE / 'predictions' / f'{today}_interval.json'
        if not path.exists():
            return False
        q_results = read_json(path)['records']
    if not q_results:
        return False
    from intraday_monitor import send_discord, WEBHOOK
    if not WEBHOOK:
        return False
    return all(send_discord(part) for part in chunks(build_message(q_results)))


if __name__ == '__main__':
    raise SystemExit('請由已授權推播流程呼叫 push_to_discord；離線預覽使用 build_message。')
