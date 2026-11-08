# API reference

The paired frontend implements OpenAI-style **Chat Completions**, with Bearer authentication when started through the supplied launcher. It is a focused subset, not full OpenAI API compatibility.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Frontend/engine status |
| `GET /telemetry` | Runtime and cache observations |
| `GET /v1/models` | The actual model names this frontend accepts |
| `POST /v1/chat/completions` | Text, reasoning, image and tool conversations; SSE or ordinary JSON |

The original launcher exposes `deepseek-v4.1-flash-dsgb10-official-1017` by default. `--model-name` changes that name. Optional steering exposes the corresponding `-normal` and `-creative` aliases instead; use `/v1/models` to discover them.

## Requests

- `messages`: system, user, assistant and tool history. Assistant `tool_calls` and corresponding tool results can be sent back on the next turn.
- `stream: true` and `stream_options: {"include_usage": true}`: SSE deltas, final usage and `[DONE]`.
- `max_tokens` or `max_completion_tokens`: positive output limit, capped by the frontend configuration. The supplied profile caps output at 393,216 tokens; input plus output must also fit the context window.
- `temperature`, `top_p`, `top_k`, `seed`: sampling controls. A seed does not promise bit-identical output across all cache states or engine versions.
- `stop`: string or string array; `stop_token_ids` and `ignore_eos` are also supported. `ignore_eos` is intended for controlled benchmarks, not ordinary agent use.
- `reasoning_effort`: `none` disables thinking; other supported effort names or numeric budgets select thinking. `chat_template_kwargs.thinking` explicitly enables/disables it. The frontend defaults to thinking off unless configured otherwise.
- `tools`: function schemas included in the model's native encoding; parsed `tool_calls` are returned. `tool_choice: "none"` omits tools. Named/required tool choices **do not provide grammar-enforced selection or schema guarantees**.
- Images: OpenAI-style `image_url` content blocks, subject to the native vision worker's count, byte and pixel limits. Text-only substitution is not used for images.
- `cache_salt`: isolates a client's prompt namespace. It does not create additional active KV capacity. `prompt_tokens_details.cached_tokens` reports actual reused input.

To disable thinking with the OpenAI SDK, use `reasoning_effort="none"`. For an explicit template option use `extra_body={"chat_template_kwargs": {"thinking": False}}`. A top-level `thinking` object is not the parser's control and should not be relied on.

## What is not implemented

There is no grammar-constrained `response_format`, JSON schema, guided regex or guided choice. The model can generate valid JSON from a prompt, but a prompt-level success is not a server-enforced constraint. Unknown fields may be ignored. `n` supports only 1. Embeddings, Responses API, audio generation and text-completions endpoints are outside this frontend.

The tool runtime belongs to the client: the server requests a function, and the client executes it and returns the result. Configure client permissions separately.

## Cache, cancellation and scheduling

Repeat the same conversation prefix to reuse its retained state. Changes to sampling, mode or image identity can select a different cache namespace. Optional steering modes keep separate histories. Actual retained KV is lost on process restart; there is no cross-restart hit guarantee.

Client disconnect cancels its request without cancelling a peer. Requests beyond available active slots wait for admission/rotation. A large cold prefill currently occupies the synchronous inference path and can delay short requests or ongoing decode. See [benchmarks](BENCHMARKS.md) for the measured effect.
