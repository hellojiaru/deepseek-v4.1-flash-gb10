# Security and privacy

This is a self-hosted inference server. The default examples bind to loopback and use a key file. Keep the API behind your normal trusted-network access controls. Model-generated tool requests are executed by your agent client, so that client remains responsible for its tool permissions.

Never attach API keys, private fleet configurations or conversations to a public issue. Use the repository's private vulnerability reporting channel when available for sensitive security reports. Ordinary reproducible bugs can use public issues with synthetic inputs.

The optional steering binary is a fixed raw float format, not executable pickle. Check its manifest and source JSON before enabling it. It changes model behavior; it does not provide an access-control boundary. See [steering](docs/STEERING.md).
