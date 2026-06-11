#!/usr/bin/env python3
"""mlx_lm.server with the --prompt-cache-bytes ceiling actually enforced.

Upstream mlx_lm (<= 0.31.x) only applies --prompt-cache-bytes on the batched
serving path, and batching is disabled whenever a draft model is loaded
(speculative decoding). On the sequential path the LRU prompt cache is
constructed without a byte limit, so it grows ~1.9GB per conversation until
the Metal GPU runs out of memory and the whole server aborts
(kIOGPUCommandBufferCallbackErrorOutOfMemory on 16GB Macs).

LRUPromptCache already evicts by max_bytes at insert time — the server just
never passes it. This wrapper patches the constructor so the cache respects
the --prompt-cache-bytes flag on every serving path, then hands control to
the stock server. Drop the wrapper once upstream enforces the flag globally.
"""

import sys

from mlx_lm.models import cache
from mlx_lm.utils import _parse_size


def _cap_from_argv(argv):
    for i, arg in enumerate(argv):
        if arg == "--prompt-cache-bytes" and i + 1 < len(argv):
            return _parse_size(argv[i + 1])
        if arg.startswith("--prompt-cache-bytes="):
            return _parse_size(arg.split("=", 1)[1])
    return None


_cap = _cap_from_argv(sys.argv)

if _cap is not None:
    _stock_init = cache.LRUPromptCache.__init__

    def _capped_init(self, max_size=10, max_bytes=1 << 63):
        _stock_init(self, max_size, min(max_bytes, _cap))

    cache.LRUPromptCache.__init__ = _capped_init

from mlx_lm.server import main

main()
