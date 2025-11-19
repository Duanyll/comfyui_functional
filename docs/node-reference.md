# ComfyUI Functional Node Reference

This document complements the [Usage Guide](./usage.md) with node-by-node details. All names match the categories that appear inside ComfyUI (`duanyll/functional/*`).

## Function Building Blocks

| Node | Description | Tips |
| --- | --- | --- |
| `Function Parameter` | Declares a positional parameter. The label and optional index determine the parameter order. | You can rename each node to clarify semantics (e.g., `mask`, `strength`). |
| `Function End` | Marks the return value of the function body. | Only one output is supported; bundle extra values into a `LIST`. |
| `Call Function` | Executes a function object and returns its output. | Connect parameters in the exact order created by Function Parameter nodes. |

## High-Order Nodes

All high-order nodes accept function objects in their first socket. They rely on Python coroutines so they can invoke your function multiple times during a single ComfyUI execution.

| Node | Signature | Use Cases | Notes |
| --- | --- | --- | --- |
| `Comap` | `(functions LIST, value)` | Compare different processing branches with the same input. | Provide a `LIST` of function objects. Each result is collected into a `LIST` in the same order. |
| `Fold` | `(function, initial_value, data LIST)` | Running accumulation (reduce). Useful for sums, chained transforms, or building strings. | Your function receives `(accumulator, element)`. |
| `Map` | `(function, data LIST)` | Apply the function to each element independently. | Returns a `LIST` with the same length as the input. |
| `MapIndexed` | `(function, data LIST)` | Like Map, but exposes the index. | Your function receives `(value, index)` with the index starting at `1`. |
| `Nest` | `(function, value, iterations INT)` | Repeat the same processing step a fixed number of times. | Equivalent to calling the function n times in a loop. |
| `NestWhile` | `(function, value, predicate, max_depth INT)` | `while` loop semantics. | After each iteration the predicate runs; if it returns `False`, iteration stops early. Use `max_depth` as a safety cap. |
| `Select` | `(predicate, data LIST)` | Filter a list according to a predicate. | Predicate must return a BOOLEAN. |
| `TakeWhile` | `(predicate, data LIST)` | Consume values until the predicate fails. | Helpful when streaming or chunking data. |

## Side-Effect Helpers

| Node | Purpose | Wiring Notes |
| --- | --- | --- |
| `Sow` | Push a value (and optional `tag`) into an implicit list. | Chain the `signal` port through any nodes that must run before the Sow executes. |
| `Reap` | Retrieve everything previously sown within the same function call. | Usually the final step of a loop. The `signal` output should continue toward downstream nodes. |
| `Inspect` | Tap into intermediate values for debugging without forcing the branch to evaluate at graph-build time. | Connect the probed value straight into the second output socket. The node only runs when its `signal` output is consumed. |
| `Sleep` | Delay execution for the specified number of seconds. | Handy when you need time to study Inspect output in the UI. |

## Locgical Utilities Nodes

| Node | Purpose |
| --- | --- |
| `IfCondition` | Standard if-then-else branching. Only selected branch executes. |
| `LogicalAnd` | Logical AND operation on boolean inputs. Short-circuits on first `False`. |
| `LogicalOr` | Logical OR operation on boolean inputs. Short-circuits on first `True`. |

## Utilities and Extensions

- **`CoroutineNodeBase`** (Python class): base class for high-order nodes. Check `comfyui_functional/high_level.py` for examples of implementing `run_coroutine`.
- **`example_workflows/`**: reference graphs for Map/Select pipelines, nested calls, recursion, and Sow/Reap debugging patterns.

## Limitations

- All nodes expect `LIST` data structures from the Basic Data Handling pack. Passing ComfyUI Data Lists will deadlock the graph.
- The scheduler hacks that make repeated execution possible may disable caching. Budget extra runtime whenever you lean on side effects.
- ComfyUI updates can invalidate assumptions about evaluation order. Test after every upgrade and file issues with reproduction workflows when possible.
