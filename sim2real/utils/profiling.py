import time
from typing import Dict, List

import torch


class ScopedTimer:
    """A context manager for timing code blocks with singleton pattern.

    Each timer name creates a singleton instance that accumulates timing data
    across multiple uses.

    Usage:
    >>> with ScopedTimer("step"):
    ...     time.sleep(1)
    >>> with ScopedTimer("step"):  # Reuses the same timer
    ...     time.sleep(1)
    >>> ScopedTimer.print_summary(clear=True)
    """

    _instances: Dict[str, "ScopedTimer"] = {}
    _stack: List["ScopedTimer"] = []
    _root_nodes: List["ScopedTimer"] = []
    children: List["ScopedTimer"] = []

    def __new__(cls, name: str, sync: bool = False):
        if ScopedTimer._stack:
            parent = ScopedTimer._stack[-1]
        else:
            parent = None
        if name not in cls._instances:
            instance = super().__new__(cls)
            instance.name = name
            instance.sync = sync
            instance.time = 0.0
            instance.count = 0
            instance.children = []
            if not parent:
                ScopedTimer._root_nodes.append(instance)
            else:
                parent.children.append(instance)
            cls._instances[name] = instance
        return cls._instances[name]

    def __init__(self, name: str, sync: bool = False):
        self.sync = sync

    def __enter__(self):
        ScopedTimer._stack.append(self)
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sync:
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        self.last_time = self.end - self.start
        self.time += self.last_time
        self.count += 1
        ScopedTimer._stack.pop()

    @staticmethod
    def print_summary(clear: bool = True):
        """Print timing summary for all timers."""
        if not ScopedTimer._instances:
            print("No timers recorded.")
            return

        print("\n" + "=" * 60)
        print(f"{'Timer Name':<30} {'Count':>8} {'Total (s)':>10} {'Avg (ms)':>10}")
        print("=" * 60)

        for root in ScopedTimer._root_nodes:
            root.print_recursive(root, 0, clear=clear)

        print("=" * 60 + "\n")

    def print_recursive(self, node: "ScopedTimer", depth: int = 0, clear: bool = True):
        """Recursively print timer nodes in DFS order."""
        indent = "  " * depth
        avg_ms = (node.time / node.count * 1000) if node.count > 0 else 0
        print(f"{indent}{node.name:<30} {node.count:>8} {node.time:>10.4f} {avg_ms:>10.2f}")

        for child in node.children:
            self.print_recursive(child, depth + 1)

        if clear:
            node.time = 0
            node.count = 0
