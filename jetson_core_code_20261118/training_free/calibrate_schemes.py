"""Freeze development-only, training-free thresholds before evaluation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from compressors import cell_difference_scores
from local_depth import calibrate_local_thresholds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    run = Path(args.run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    requests = {r['request_id']: r for r in json.loads((run/'requests.json').read_text())}
    rows = [json.loads(s) for s in (run/'results.jsonl').read_text().splitlines()]
    records = []
    differences = []
    for row in rows:
        if row['arm'] != 'skip_local' or row['phase'] != 'measure':
            continue
        request = requests[row['request_id']]
        records.append({'video_id': request['video_id'], 'features': row['stats']['calls'][0]['features']})
        case = torch.load(Path(args.features)/(row['request_id']+'.pt'), map_location='cpu', weights_only=True)
        t, h, w = case['grid_thw'][0]
        differences.append(cell_difference_scores(case['x'][0], (t,h//2,w//2)))
    thresholds = calibrate_local_thresholds(records)
    difference = float(torch.quantile(torch.cat(differences).float(), .5))
    (out/'local_thresholds.json').write_text(json.dumps(thresholds, indent=2)+'\n')
    (out/'compression_threshold.json').write_text(json.dumps({'difference_threshold': difference}, indent=2)+'\n')
    (out/'receipt.json').write_text(json.dumps({'source': str(run), 'records': records,
        'rule': 'Local depth marginal q75 for skip1, q50 for skip2; both conditions required. Compression cell relative variance pooled q50.',
        'num_compression_cells': sum(x.numel() for x in differences),
        'no_training': True, 'answer_labels_used': False}, indent=2)+'\n')
    print(json.dumps({'local': thresholds, 'difference_threshold': difference}))


if __name__ == '__main__':
    main()
