# Decision Scorecard

| Criterion | Mixed-V kernel | Cache layout |
|---|---|---|
| % measured time | largest component at 32768; CUDA mean `1493.893` us/call | second-largest systems component; `35904160` estimated bytes/token |
| scaling with T | mixed CUDA grows from 16K to 32K but remains kernel-dominated | bytes/token ratio 16K->32K `1.172` (constant) |
| bytes moved | compact temp allocation reported in `mixed_v_temp_allocations.csv` | top bytes category `recent_pending` |
| launch overhead | two CUDA launches per mixed call when both lanes present | many small concat/mutation events per token |
| optimization headroom | likely inside existing V2/V4 CUDA kernel and two-lane launch structure | ABI change could remove dynamic copies but does not target rank-1 compute |
| relevance to vLLM | kernel remains relevant under any scheduler | fixed pages are relevant later for vLLM-style allocators |
| implementation risk | moderate; kernel-level optimization with existing ABI | high; storage/page ABI change touches cache semantics broadly |

Decision: `S2B_MIXED_V_KERNEL_OPTIMIZATION`

Reason: Mixed-V CUDA execution is the largest remaining root cause; host/layout overhead is small, and cache copy bytes/token growth from 16K to 32K is 1.172 (constant).
