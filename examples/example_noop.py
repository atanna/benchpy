# -*- coding: utf-8 -*-

import benchpy as bp


def noop():
    pass


if __name__ == "__main__":
    print(bp.run(bp.bench("noop", noop),
                 n_samples=4, max_batch=32, n_batches=4))
