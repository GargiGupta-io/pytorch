# Case Study: Deterministic Ordering in `get_source_partitions`

This document explains a bug in PyTorch's FX graph utilities, why it occurred, and how it was fixed. It is written for developers who may be new to Python data structures, FX graphs, or quantization backends.

## Table of Contents

1. [Background: What You Need to Know](#background-what-you-need-to-know)
2. [The Bug in Context](#the-bug-in-context)
3. [Root Cause: Non-Deterministic Ordering](#root-cause-non-deterministic-ordering)
4. [The Fix](#the-fix)
5. [Key Takeaways](#key-takeaways)

---

## Background: What You Need to Know

### What is PyTorch FX?

FX (Function eXecutor) is PyTorch's toolkit for working with models as **graphs**. Instead of running Python code directly, FX converts a `nn.Module` into a graph of operations (nodes) and data flow (edges). This graph can be inspected, transformed, and optimized—for example, for quantization.

### What is `get_source_partitions`?

`get_source_partitions` is a function in `torch.fx.passes.utils.source_matcher_utils` that scans an FX graph and finds groups of nodes that came from a given "source"—such as a Python function like `torch.gather` or a module type like `nn.Linear`. Each group is returned as a `SourcePartition` containing:

- **`nodes`**: The nodes that belong to this partition
- **`input_nodes`**: Nodes *outside* the partition that feed into it (e.g., inputs and constants)
- **`output_nodes`**: Nodes in the partition whose outputs are used outside the partition
- **`params`**: Parameter nodes (e.g., `get_attr` for weights)

For operators with multiple inputs (like `torch.gather`), `input_nodes` may contain both the data tensor and the index tensor. **The order of items in `input_nodes` matters** when downstream code assumes a fixed mapping (e.g., "the first input is the tensor to quantize").

### Python Sets vs. Dicts: A Quick Primer

- **`set`**: An unordered collection of unique items. Iteration order is not guaranteed and can vary between runs.
- **`dict`** (Python 3.7+): A mapping that preserves **insertion order**. The order in which keys are added is the order in which they are returned when iterating.

Using a set for `input_nodes` and then converting it to a list with `list(input_nodes)` produces a list whose order is undefined. Using a dict instead (or a list with duplicate checks) preserves the order in which nodes were first encountered.

---

## The Bug in Context

### Original Use Case

A user was extending PyTorch's quantization backend for PT2E (PyTorch 2.0 Export). They used `get_source_partitions` to find `torch.gather` nodes and annotate them for quantization. Their code assumed that the input tensor (to be quantized) was always at a fixed index in `input_nodes`:

```python
input_node = input_nodes[1]  # Assumed: index 1 is always the input tensor
input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
```

### The FX Graph

After export, the graph looked like this:

```
%gather = call_function[target=torch.ops.aten.gather.default](args = (%x, 0, %detach), kwargs = {})
```

- `%x`: The placeholder (user input) — this is the tensor to quantize
- `0`: The dimension
- `%detach`: The index tensor (from a lifted constant)

### Observed Behavior

The user ran the same code 100 times. In about 25 runs, the input tensor `%x` appeared at `input_nodes[1]`; in the other ~75 runs, it appeared at `input_nodes[0]`. Their assumption about `input_nodes[1]` was wrong roughly 75% of the time, causing incorrect quantization and flaky tests.

---

## Root Cause: Non-Deterministic Ordering

### The Problematic Code (Before Fix)

In `torch/fx/passes/utils/source_matcher_utils.py`, the inner function `make_partition` built `input_nodes`, `output_nodes`, and `params` using Python **sets**:

```python
def make_partition(nodes: list[Node], module_type: type) -> SourcePartition:
    input_nodes = set()
    output_nodes = set()
    params = set()
    for node in nodes:
        for arg in node.args:
            if isinstance(arg, Node) and arg not in nodes and arg.op != "get_attr":
                input_nodes.add(arg)
        # ... similar for output_nodes and params ...

    return SourcePartition(
        nodes,
        module_type,
        list(input_nodes),   # ← Order is undefined!
        list(output_nodes),
        list(params),
    )
```

### Why the Order Was Inconsistent

1. **Sets are unordered**: A Python `set` does not guarantee any iteration order.
2. **`list(set)` is arbitrary**: Converting a set to a list yields an order that depends on the internal hash values of the elements.
3. **`Node` hash values vary**: `Node` objects use identity-based hashing. Their hash values can change between runs due to memory layout, allocation order, or interpreter behavior.
4. **Result**: `list(input_nodes)` could be `[x, detach]` in one run and `[detach, x]` in another.

For `torch.gather`, both `%x` and `%detach` are added to `input_nodes`. Because the set's iteration order was non-deterministic, the list passed to `SourcePartition` had no stable ordering.

---

## The Fix

### Strategy

Use **insertion-ordered collections** instead of sets so that the order in which nodes are first encountered is preserved. In Python 3.7+, a `dict` is insertion-ordered, so we can use it like an ordered set by storing `None` as the value.

### Exact Code Changes

**File:** `torch/fx/passes/utils/source_matcher_utils.py`

**Before:**

```python
def make_partition(nodes: list[Node], module_type: type) -> SourcePartition:
    input_nodes = set()
    output_nodes = set()
    params = set()
    for node in nodes:
        for arg in node.args:
            if isinstance(arg, Node) and arg not in nodes and arg.op != "get_attr":
                input_nodes.add(arg)

        if node.op == "get_attr":
            params.add(node)
            continue

        for user in node.users:
            if user not in nodes:
                output_nodes.add(node)

    return SourcePartition(
        nodes,
        module_type,
        list(input_nodes),
        list(output_nodes),
        list(params),
    )
```

**After:**

```python
def make_partition(nodes: list[Node], module_type: type) -> SourcePartition:
    input_nodes: dict[Node, None] = {}
    output_nodes: dict[Node, None] = {}
    params: dict[Node, None] = {}
    nodes_set = set(nodes)
    for node in nodes:
        for arg in node.args:
            if isinstance(arg, Node) and arg not in nodes_set and arg.op != "get_attr":
                input_nodes.setdefault(arg)

        if node.op == "get_attr":
            params.setdefault(node)
            continue

        for user in node.users:
            if user not in nodes_set:
                output_nodes.setdefault(node)
                break

    return SourcePartition(
        nodes,
        module_type,
        list(input_nodes),
        list(output_nodes),
        list(params),
    )
```

### Line-by-Line Explanation

| Change | Reason |
|--------|--------|
| `input_nodes = set()` → `input_nodes: dict[Node, None] = {}` | Use a dict so insertion order is preserved. We only care about keys; values are `None`. |
| `output_nodes = set()` → `output_nodes: dict[Node, None] = {}` | Same as above for output nodes. |
| `params = set()` → `params: dict[Node, None] = {}` | Same as above for parameter nodes. |
| `nodes_set = set(nodes)` | Build a set once for O(1) membership checks. Avoids repeated `arg not in nodes` on a list, which is O(n). |
| `arg not in nodes` → `arg not in nodes_set` | Use the precomputed set for faster lookups. |
| `input_nodes.add(arg)` → `input_nodes.setdefault(arg)` | `setdefault` adds the key if missing and preserves insertion order. Does not overwrite if the key already exists. |
| `params.add(node)` → `params.setdefault(node)` | Same as above for params. |
| `output_nodes.add(node)` → `output_nodes.setdefault(node)` + `break` | Add the node once and stop; no need to keep iterating over other users. The `break` is a small optimization. |

### Why the Order Is Now Deterministic

- **`input_nodes`**: Nodes are added in the order they appear in `node.args`, as we traverse `nodes` and their arguments. That order is fixed by the graph structure.
- **`output_nodes`**: Nodes are added in the order they appear in `nodes`, when we first see that a node has an external user.
- **`params`**: Nodes are added in the order `get_attr` nodes appear in `nodes`.

For `aten.gather.default(%x, 0, %detach)`:

- `node.args` is `(x, 0, detach)`
- We process `x` first, then `detach` (skipping the scalar `0`)
- So `input_nodes` is always `[x, detach]` in a deterministic order.

---

## Key Takeaways

1. **Do not rely on set iteration order** when the order matters. Use an ordered structure (e.g., `dict` in Python 3.7+, or a list with duplicate checks).
2. **`SourcePartition.input_nodes` order now follows graph structure** (argument order), so consumers can depend on it.
3. **Prefer `set` for membership checks** when you need to test `x in collection` many times; a set gives O(1) lookups vs. O(n) for a list.
4. **For quantization and similar backends**, use the node’s `args` or `kwargs` to identify which input is which (e.g., `gather_node.args[0]` for the input tensor), rather than assuming a fixed index in `input_nodes`. Relying on deterministic `input_nodes` order is now safe, but argument-based lookup is more robust and self-documenting.

---

## Related Code and Tests

- **Fixed file:** `torch/fx/passes/utils/source_matcher_utils.py`
- **Tests:** `test/fx/test_source_matcher_utils.py`
