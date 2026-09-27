"""Gate exact-output APPS rewards on sandbox audit and unambiguous statements."""

import json
import re
from pathlib import Path

from .core import digest


def code_exclusions(rows, audit_dir):
    if not audit_dir:
        raise ValueError('APPS requires MIXRL_CODE_AUDIT_DIR; run the sandbox reference audit first')
    root = Path(audit_dir)
    summary = json.loads((root / 'summary.json').read_text())
    apps = [r for r in rows if r['task'] == 'apps']
    if summary['total'] != len(apps):
        raise ValueError('APPS audit is incomplete for the frozen train/val splits')
    excluded, identities = {}, {}
    for row in apps:
        record = json.loads((root / (row['id'] + '.json')).read_text())
        if record['row_hash'] != digest(row):
            raise ValueError('APPS audit row identity changed')
        identities[row['id']] = digest(record)
        if record.get('error'):
            raise ValueError('APPS audit has an infrastructure failure; repair/retry it')
        if not record.get('passed'):
            excluded[row['id']] = 'No published reference solution passed the sandbox verifier'
            continue
        negative = record.get('negative', {})
        if negative.get('status') != 'valid' or negative.get('passed') is not False:
            raise ValueError('APPS audit lacks a valid negative control')
        prompt = '\n'.join(m['content'] for m in row['messages']).casefold()
        if re.search(r'\b(?:print|output)\s+any\b|\bin any order\b|'
                     r'\b(?:multiple|several|more than one)\s+(?:valid\s+)?(?:answers?|solutions?)\b', prompt):
            excluded[row['id']] = 'Potential non-unique output requires a task-specific checker; conservative exclusion'
    return excluded, digest(identities)
