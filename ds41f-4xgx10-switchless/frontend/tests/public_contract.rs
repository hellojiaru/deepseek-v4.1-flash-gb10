//! Public, synthetic CPU contracts. No checkpoint, socket, inference or credentials.
use dsfront::encoding::{Encoding, IMAGE_PLACEHOLDER};
use dsfront::server::{prepare, ServerConfig, VisionPrefixPolicy};
use serde_json::{json, Value};
use std::time::Duration;

fn config() -> ServerConfig {
    ServerConfig {
        creative: true, vision_prefix_policy: VisionPrefixPolicy::TailExact,
        cache_scope: "synthetic-revision:synthetic-assets".into(),
        vision_socket: Some("unused-cpu-fixture".into()),
        vision_revision: Some("synthetic-vision-revision".into()),
        model_name: "public-test".into(), api_key: None, max_tokens_cap: 393216,
        default_effort: 75, encoding: Encoding::V41_1017, default_thinking: false,
        default_temperature: 1.0, default_top_p: 1.0, loop_guard: None,
        tokenize_threads: 1, dump_dir: None, hold: Duration::from_secs(1),
    }
}
fn text_request() -> Value {
    json!({"model":"public-test-normal", "cache_salt":"synthetic",
           "messages":[{"role":"user","content":"Explain the synthetic test."}]})
}
fn picture(url: &str) -> Value {
    json!({"type":"image_url","image_url":{"url":url}})
}

#[test]
fn policy_and_mode_namespace_never_alias() {
    let request = text_request();
    let mut cfg = config();
    let normal = prepare(&cfg, &request).unwrap();
    let mut creative_request = request.clone();
    creative_request["model"] = json!("public-test-creative");
    let creative = prepare(&cfg, &creative_request).unwrap();
    assert_ne!(normal.cache_namespace, creative.cache_namespace);
    assert!(!normal.creative && creative.creative);
    for policy in [VisionPrefixPolicy::Exact, VisionPrefixPolicy::Legacy] {
        cfg.vision_prefix_policy = policy;
        assert_ne!(normal.cache_namespace, prepare(&cfg, &request).unwrap().cache_namespace);
    }
}

#[test]
fn revision_and_encoding_namespace_never_alias() {
    let request = text_request();
    let mut cfg = config();
    let baseline = prepare(&cfg, &request).unwrap();
    cfg.cache_scope.push_str(":next-assets");
    assert_ne!(baseline.cache_namespace, prepare(&cfg, &request).unwrap().cache_namespace);
    cfg = config(); cfg.encoding = Encoding::V41;
    assert_ne!(baseline.cache_namespace, prepare(&cfg, &request).unwrap().cache_namespace);
}

#[test]
fn tool_result_picture_order_follows_calls_not_arrival() {
    // These strings are identity fixtures, not image decoder inputs.
    let first = "data:image/png;base64,Zmlyc3Q=";
    let second = "data:image/png;base64,c2Vjb25k";
    let request = json!({"messages":[
        {"role":"user","content":"Read both synthetic pictures."},
        {"role":"assistant","content":null,"tool_calls":[
            {"id":"first","type":"function","function":{"name":"capture","arguments":"{}"}},
            {"id":"second","type":"function","function":{"name":"capture","arguments":"{}"}}
        ]},
        {"role":"tool","tool_call_id":"second","content":null,"content_blocks":[picture(second)]},
        {"role":"tool","tool_call_id":"first","content":null,"content_blocks":[picture(first)]}
    ]});
    let prepared = prepare(&config(), &request).unwrap();
    // The canonical encoder records retain their image type as well as source URL.
    assert_eq!(prepared.images, vec![
        json!({"type":"image", "url":first}),
        json!({"type":"image", "url":second}),
    ]);
    assert_eq!(prepared.prompt.matches(IMAGE_PLACEHOLDER).count(), 2);
}

#[test]
fn missing_vision_worker_is_explicit_error() {
    let request = json!({"messages":[{"role":"user","content":[picture("data:image/png;base64,AA==")]}]});
    let mut cfg = config(); cfg.vision_socket = None;
    assert!(prepare(&cfg, &request).err().unwrap().contains("vision encoder is not configured"));
}

#[test]
fn budgets_are_bounded_without_silent_sampling_changes() {
    let mut request = text_request(); request["max_tokens"] = json!(393217);
    let p = prepare(&config(), &request).unwrap();
    assert_eq!(p.max_tokens, 393216);
    assert_eq!((p.temperature, p.top_p, p.top_k), (1.0, 1.0, 0));
    request["max_tokens"] = json!(0);
    assert!(prepare(&config(), &request).is_err());
    request["max_tokens"] = json!(16); request["cache_salt"] = json!("x".repeat(4097));
    assert!(prepare(&config(), &request).is_err());
}

#[test]
fn unknown_policy_is_rejected_and_default_is_not_redefined() {
    assert!(VisionPrefixPolicy::parse("approximate").is_err());
    assert!(!VisionPrefixPolicy::Legacy.shared_image_prefix());
    assert!(VisionPrefixPolicy::Exact.shared_image_prefix());
    assert!(VisionPrefixPolicy::TailExact.shared_image_prefix());
    assert_ne!(VisionPrefixPolicy::Exact.flags(), VisionPrefixPolicy::TailExact.flags());
}
