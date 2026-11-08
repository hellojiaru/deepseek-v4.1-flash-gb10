# Benchmarks and operating evidence

All local measurements use DeepSeek-V4.1-Flash-1017 on four GB10 nodes unless a report says otherwise. Each report separates its workload, cache state, timing boundary and evidence scope.

| Question | Result | Report |
|---|---|---|
| What does a normal long agent task feel like? | 8 ordinary tasks, 2.55M output tokens; long-response median 108.7 tok/s, all-response weighted 103.4 | [Historical ordinary agents](benchmarks/ordinary-agents-20261109.md) |
| Does the original, unsteered launcher work? | Actual four-rank launch, 24/24 outputs equal to both controls; tools, image and 65K cache pass | [Original launcher](benchmarks/original-launch-20261109.md) |
| How fast is uncached input? | Three samples each; 65K mean9,281 and 262K mean9,611 input tok/s | [Prefill and interference](benchmarks/prefill-and-latency-20261109.md) |
| Does a short request wait behind a large prefill? | A short request arriving 0.76s after a cold 262K input waited 26.72s to first content | [Same report](benchmarks/prefill-and-latency-20261109.md) |
| What are C1/C4 generation speeds? | C1 code 142.2/prose 94.1; C4 aggregate 173.4/152.6 tok/s | [Paired source build](integration/VALIDATION_20261108.md) |
| What happens with two requests? | C2 aggregate code 148.8/prose 127.6 tok/s; three repeats | [C2 reference](benchmarks/c2-20261109.md) |
| Does C8 really batch eight active requests? | C8 aggregate 212.2/180.8; actual native eight-request batches and stream overlap | [C8 reference](integration/C8_REFERENCE_20261108.md) |
| Do eight 1M contexts fit and generate? | Eight × (1,048,320 input+256 output), 24/24 markers, no new swap/OOM kill/spill | [Full8M capacity](benchmarks/capacity-8m-20261108.md) |
| Is there longer operating history? | Previous C4 process uptime 123.6h; separate recent workload cohort 3.97M output tokens | [Production history](benchmarks/real-workloads-20261108.md) |

**Generation speed** measures output after the first stream content; it excludes tool execution and human pauses. **Effective prefill rate** divides actual input tokens by request-to-first-content time. **Concurrent aggregate speed** divides the group's output by a common time interval; it is not one user's speed multiplied by the concurrency setting.

The capacity input is a synthetic repeated pattern with retrieval markers. Its success proves the tested memory/window behavior and marker recall, not comprehensive 1M quality. Historical agent performance comes from real varying tasks and historical builds; it is not an isolated current-build comparison. Uptime includes idle periods.

Raw private prompts, session titles, paths and tool bodies are not distributed. Numerical summaries, hashes and reproducible synthetic/public-input probes accompany the reports. [Competitor comparison](COMPARISON.md) uses author-published figures with their conditions, without claiming matched hardware runs.
