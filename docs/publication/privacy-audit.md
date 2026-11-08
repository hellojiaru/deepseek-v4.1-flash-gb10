# Publication privacy boundary

Only implementation sources, generated/synthetic fixtures, dependency metadata,
and authored documentation belong in this contribution. Private game prompts,
image histories, reasoning text, request/response captures, account records,
credentials, host inventories, and deployment transaction directories do not.

The measurement summary intentionally includes aggregate counts/timing and
limitations, not the private input text or raw operational records. Existing
upstream benchmark fixtures are retained under their original notices; they
must not be confused with newly contributed private workloads.

Before publication, inspect every added file and added diff line, include
untracked files in the inventory, verify referenced license/source copies, and
search for secrets, personal filesystem roots, deployment addresses, and raw
request/session artifacts. Search results require manual classification: a
synthetic token sequence, public source hash, or an upstream-existing fixture is
not automatically private data. Conversely, a successful string scan is not a
proof that all content is safe to redistribute.

## Review of this import, 2026-10-21

The review covered added lines of tracked changes and the full contents of new
files relative to upstream `a92b8f44046f4435c3c29c07fbc39e6e4b157080`, including
engine/frontend tests, vision sources, public smoke fixtures, license files,
and CPU workflow configuration. Source/reference hashes and local documentation
links were checked. No newly introduced credentials, private user prompts,
private image binaries, account records, session traces, personal filesystem
roots, or deployment network addresses were found in that scope.

Two initial path-pattern hits were manually classified as a `file://` rejection
comment and a generated `file:///synthetic.png` rejection test. They are not
local data or private paths. The public smoke input is generated; its bearer key
is read only from the caller's environment when HTTP execution is explicitly
requested and is not recorded in its result.

Existing upstream historical fixtures, example topology, and reference material
were not removed or represented as newly audited private data. This is a source
publication review, not a general security assessment of URL fetching, remote
execution, upstream dependencies, or all future contributions. CPU tests do not
establish GPU compatibility or reproduce the historical deployment benchmark.

The accompanying [review inventory](privacy-audit.json) records the reviewed
file hashes. It excludes itself to avoid a self-referential hash. Changes after
that inventory require their own review; it does not authorize publishing
arbitrary local evidence or future unreviewed files.
