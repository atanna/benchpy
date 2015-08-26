# -*- coding: utf-8 -*-

import benchpy as bp


def cyclic_list(n):
    for i in range(n):
        cycle = []
        cycle.append(cycle)


if __name__ == "__main__":
    n_cycles = 128
    name = "cycle_list({})".format(n_cycles)
    groups = [
        bp.group("+GC", [bp.bench(name, cyclic_list, n_cycles)], with_gc=True),
        bp.group("-GC", [bp.bench(name, cyclic_list, n_cycles)], with_gc=False)
    ]

    print(bp.run(groups, n_samples=16, max_batch=32, n_batches=10, n_jobs=-1))
