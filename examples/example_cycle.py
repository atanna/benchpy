# -*- coding: utf-8 -*-

import benchpy as bp


def cyclic_list(n):
    cycle = []
    for i in range(n):
        cycle.append(cycle)


if __name__ == "__main__":
    n_cycles = 8
    groups = [
        bp.group("+GC",
                 [bp.bench("cyclic_list(8)", cyclic_list, n_cycles)],
                 with_gc=True),
        bp.group("-GC",
                 [bp.bench("cyclic_list(8)", cyclic_list, n_cycles)],
                 with_gc=False)
    ]

    print(bp.run(groups, n_samples=4, max_batch=32, n_batches=4))
