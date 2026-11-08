# Contributing

Start with the [build guide](docs/BUILD_AND_RUN.md) and [maintenance checks](docs/MAINTENANCE.md). Explain the user-visible change and keep source attribution intact. Run the public CPU tests before submitting a pull request.

For performance changes, provide the checkpoint revision, source and executable hashes, hardware/topology, launch arguments, prompts, cache state and repeat counts. Report single-request and aggregate throughput separately; include TTFT and failures. Preserve image, tool, cache and original-model behavior when claiming a drop-in improvement. GPU tests require an explicitly available, bounded four-node environment; CI does not reserve one.

Do not submit credentials, real fleet files, private conversations, captures or API dumps. Use synthetic fixtures. Optional direction assets need provenance, license, readable values, hashes and an explicit opt-in path. Public release artifacts must follow [the publication checks](docs/PUBLICATION.md).
