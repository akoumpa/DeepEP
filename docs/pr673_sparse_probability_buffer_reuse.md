# PR #673 sparse probability buffer reuse defect

## Summary

DeepEP PR [#673](https://github.com/deepseek-ai/DeepEP/pull/673) can expose stale
probability values from the preceding `combine()` call through the non-permute
`dispatch()` API when `HybridEPBuffer` uses its default shared-buffer mode.

The affected path writes only the destination rank's `E_per_rank` probability
slice but returns a dense `E_per_rank * ranks_per_node` tensor. The columns not
written by the current dispatch retain data already present in the shared
combine/dispatch buffer.

This is deterministic buffer reuse, not a stochastic floating-point difference.

## Root cause

Commit `62f7232` optimized forward probability communication by copying only the
destination rank's expert slice. Commit `db9880b` added a full-buffer zero-fill
option in `Executor::dispatch_core`, but left it compile-time disabled:

```cpp
static constexpr bool kZeroDispatchProbBufferBeforeSparseWrites = false;
```

Three implementation details combine to expose the stale values:

1. With `use_shared_buffer=true`, `dispatch_buffers.expert_output_prob` aliases
   `combine_buffers.expert_input_prob`.
2. A non-permute `combine()` copies its complete dense probability input into
   that shared allocation.
3. The following sparse `dispatch()` replaces only one rank-local slice, after
   which `dispatch_postprocess()` copies the complete dense allocation into the
   returned tensor.

## Reproduction

`tests/test_hybrid_ep.py` contains an opt-in regression check. It sends a
`12345.0` sentinel through the public `combine()` API before issuing another
sparse `dispatch()`:

```bash
python tests/test_hybrid_ep.py \
  --num-processes 8 \
  --only-bf16 \
  --correctness-only \
  --assert-dense-prob-zero-fill
```

On the exact PR #673 head (`936e059`), every off-slice element checked retained
the sentinel. Counts ranged from `1,223,936 / 1,223,936` to
`1,263,808 / 1,263,808` per rank on an 8x H100 node. The command fails with:

```text
AssertionError: Dispatch probs retained ... poisoned off-slice values from combine
```

On a build with the sparse probability commits reverted, all eight ranks report
zero retained sentinel values and the check passes.

## Scope and numerical impact

The raw non-permute `dispatch()` API violates the prior dense-output behavior.
A caller that reads the complete returned probability tensor can consume values
from an earlier operation.

The `dispatch_with_permute()` path selects mapped local-expert entries instead
of returning untouched dense columns. The same shared-buffer poison sequence was
tested against both fused and non-fused permute implementations; hidden states,
probabilities, and scaling factors remained bitwise equal to the reference.
That is the path currently used by NeMo AutoModel's HybridEP dispatcher.

This defect therefore does not explain the transient Qwen3-Next benchmark loss
change from `12.2530` to `12.3155`. That difference is one BF16 bin after the
benchmark's loss sum is divided by `2,096,640` labels, and it also occurs on
builds without the PR #673 sparse probability changes.

## Remediation options

- Restore zero-initialization before sparse writes when the non-permute API must
  preserve a dense zero-filled output contract.
- Zero the newly allocated public output tensor and copy only the valid local
  expert slice during non-permute postprocessing, avoiding a memset of the
  shared communication buffer.
- Explicitly narrow the non-permute API contract to expose only the valid slice
  and update callers and tests accordingly.

Whichever contract is chosen should have a combine-to-dispatch shared-buffer
regression test so ordinary zero-initialized test data cannot hide stale values.
