"""Preserve raw rows; summarize only complete request/repeat sets without contamination."""
import argparse
import json
from pathlib import Path
from summarize_schemes import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('results')
    p.add_argument('--ignore-foreign-pid', type=int, action='append', default=[])
    args = p.parse_args()
    path = Path(args.results)
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    ignored = set(args.ignore_foreign_pid)
    bad = {(r['request_id'], r['repeat']) for r in rows if r['phase']=='measure'
           and any(x['pid'] not in ignored for x in r['hardware']['foreign_processes'])}
    report = summarize([r for r in rows if (r['request_id'], r['repeat']) not in bad])
    report['excluded_pairs'] = [list(k) for k in sorted(bad)]
    report['ignored_foreign_pids'] = sorted(ignored)
    report['filter_note'] = ('Exclude the whole request/repeat across all arms if any measurement '
        'observed another foreign process. Ignored PIDs must be identified explicitly, e.g. Xorg. '
        'Ignoring display processes is not proof of exclusive GPU or host resources. Raw rows unchanged.')
    output = path.parent/'summary_filtered.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
