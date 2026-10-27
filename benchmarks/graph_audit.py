"""CUDA graph DOT inspection without numerical or comparison dependencies."""

import collections
import re
from pathlib import Path


def graph_summary(path):
    text = Path(path).read_text()
    return {
        "node_types_including_conditional_body": dict(
            collections.Counter(re.findall(r'label="\{\s*([A-Z_]+)', text))
        ),
        "memcpy_kinds": dict(collections.Counter(re.findall(r"\| \{kind \| ([^}]+)", text))),
        "method": "CUDA graph DOT inspection; not a CUDA API trace",
    }
