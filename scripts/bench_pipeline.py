#!/usr/bin/env python3
"""Interleave serial/pipeline runs; use elapsed time for speedup, perf for work.

Build first: cargo build --release -p zfp-benchmarks --example pipeline
Default CPU order is for the measurement laptop (four P-, then four E-cores).
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='target/release/examples/pipeline')
    parser.add_argument('--output', type=Path, default=Path('/tmp/zfp-pipeline-results'))
    parser.add_argument('--cpus', default='3,1,0,2,4,5,6,7')
    parser.add_argument('--threads', type=int, nargs='+', default=[2, 4, 8])
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch', type=int, default=0)
    parser.add_argument('--fresh', action='store_true', help='create a pool on every decode')
    parser.add_argument('--cases', nargs='+', help='exact case names; default all 56')
    parser.add_argument('--events', default='cpu_core/cycles/u,cpu_core/instructions/u,cpu_atom/cycles/u,cpu_atom/instructions/u')
    args = parser.parse_args()
    assert args.iterations > 0 and args.repeats > 0
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ZFP_BENCH_CPUS=args.cpus)
    cases = args.cases or [f'{t}_d{d}_{m}' for t in ['i32', 'i64', 'f32', 'f64']
                          for d in range(1, 5)
                          for m in ['fixed_rate', 'fixed_precision', 'fixed_accuracy', 'reversible']
                          if m != 'fixed_accuracy' or t.startswith('f')]
    rows = []

    def sample(case, threads, iterations, tag):
        path = args.output / f'{case}.t{threads}.{tag}.perf'
        command = ['taskset', '-c', args.cpus.split(',')[0], 'perf', 'stat', '--no-scale', '-x,',
                   '-e', args.events, '-o', str(path), args.binary,
                   case, str(iterations), str(threads), str(args.batch)]
        if args.fresh:
            command.append('fresh')
        proc = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
        ns = int(re.search(r'\bns=(\d+)', proc.stdout)[1])
        counts = {}
        for row in csv.reader(path.read_text().splitlines()):
            if len(row) > 4 and row[2] in args.events.split(','):
                # A hybrid PMU not used by any thread is legitimately uncounted.
                counts[row[2]] = float(row[0]) if row[0].strip().isdigit() else 0
        assert counts and counts.get('cpu_core/instructions/u', 0) > 0, path
        return ns, counts

    for index, case in enumerate(cases):
        for rep in range(args.repeats):
            # A,B2,A,B4,A,B8 on each repetition; each sample has its own zero run.
            for threads in args.threads:
                for selected in [0, threads]:
                    tag = f'r{rep}.pair{threads}'
                    _, zero = sample(case, selected, 0, tag + '.zero')
                    ns, total = sample(case, selected, args.iterations, tag + '.timed')
                    counters = {k: (total[k] - zero.get(k, 0)) / args.iterations for k in total}
                    assert counters.get('cpu_core/instructions/u', 0) > 0
                    rows.append(dict(case=case, threads=selected, paired_threads=threads,
                                     repeat=rep, batch=args.batch, fresh=args.fresh,
                                     iterations=args.iterations, ns=ns / args.iterations,
                                     **counters))
        (args.output / 'raw.json').write_text(json.dumps(rows, indent=2) + '\n')
        speedups = {}
        for threads in args.threads:
            pair = [r for r in rows if r['case'] == case and r['paired_threads'] == threads]
            serial = min(r['ns'] for r in pair if r['threads'] == 0)
            parallel = min(r['ns'] for r in pair if r['threads'] == threads)
            speedups[threads] = round(serial / parallel, 3)
        print(f'{index+1}/{len(cases)} {case} {speedups}', flush=True)
    with (args.output / 'raw.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'settings.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')


if __name__ == '__main__':
    main()
