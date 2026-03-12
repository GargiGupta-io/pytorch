# Case Study Update: Tests for Deterministic Ordering

This document supplements [case_study_source_partition_ordering.md](case_study_source_partition_ordering.md) with the tests added to verify the fix.

## Overview

Two tests were added to ensure `get_source_partitions` returns deterministic `input_nodes` order:

1. **Official test** (runs in PyTorch CI)
2. **Standalone test** (for environments where the full test suite cannot run)

---

## Official Test

**File:** `test/fx/test_source_matcher_utils.py`  
**Test:** `test_get_source_partitions_input_nodes_deterministic_order`

### What It Verifies

1. **Order matches `node.args`**: For `torch.gather(input, dim, index)`, `input_nodes` must be `[input, index]` in that order.
2. **Determinism**: Running `get_source_partitions` 10 times on the same graph yields identical `input_nodes` order each time.

### How to Run

With a built PyTorch from source:

```bash
python test/fx/test_source_matcher_utils.py TestSourceMatcher.test_get_source_partitions_input_nodes_deterministic_order -v
```

**Note:** Run from a directory where the pytorch repo is not on `PYTHONPATH`, or ensure you use the built/installed PyTorch, not the raw source.

---

## Standalone Test

**File:** `test_source_partition_ordering_standalone.py` (repository root)

### Why It Exists

The official test imports from `torch.testing._internal.common_utils` (`raise_on_run_directly`, etc.), which is only available in a full PyTorch development setup. If you have PyTorch installed via `pip` or `conda` but not built from source, the official test may fail with import errors.

The standalone test uses only the public `torch` API and `get_source_partitions`, so it runs with any installed PyTorch.

### What It Verifies

Same as the official test: order matches `node.args`, and order is deterministic across repeated calls.

### How to Run

Run from **outside** the pytorch repo so the installed PyTorch is used:

```bash
cd /path/to/your/home
python /path/to/pytorch/test_source_partition_ordering_standalone.py
```

On Windows:

```powershell
cd C:\Users\HP
python C:\Users\HP\pytorch\test_source_partition_ordering_standalone.py
```

### Expected Output

```
OK - input_nodes order is deterministic and matches node.args
```

You may see `UserWarning` messages about lifted tensors; those are benign and unrelated to the ordering fix.

---

## Test Logic (Both Tests)

```python
# 1. Build a model with torch.gather
class GatherLayer(nn.Module):
    def forward(self, x):
        return torch.gather(x, dim=0, index=torch.tensor([[0, 0], [1, 0]], device=x.device))

# 2. Export to FX
gm = torch.export.export(GatherLayer(), example_inputs).module()

# 3. Get partitions and verify order
partitions = get_source_partitions(gm.graph, [torch.gather])
partition = partitions[torch.gather][0]
gather_node = partition.output_nodes[0]

# input_nodes must follow node.args order (input first, then index)
input_node_order = [n.name for n in partition.input_nodes]
args_node_order = [arg.name for arg in gather_node.args if isinstance(arg, fx.Node)]
assert input_node_order == args_node_order

# 4. Verify determinism
for _ in range(10):
    partition_repeat = get_source_partitions(gm.graph, [torch.gather])[torch.gather][0]
    assert [n.name for n in partition_repeat.input_nodes] == input_node_order
```

---

## Related Files

| File | Purpose |
|------|---------|
| `torch/fx/passes/utils/source_matcher_utils.py` | The fix |
| `test/fx/test_source_matcher_utils.py` | Official test (CI) |
| `test_source_partition_ordering_standalone.py` | Standalone test |
| `docs/learning/case_study_source_partition_ordering.md` | Bug and fix documentation |
