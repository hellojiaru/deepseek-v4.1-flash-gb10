//! Numeric-only, process-epoch counters. No text, token IDs, credentials or disk I/O.
//! Output follows Gen.kept (including its first/stop token), not SSE chunks or drafts.
use crate::protocol::{Done, DoneReason};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Default, Debug, Serialize)]
pub struct Counters {
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub output_tokens: u64,
    pub requests_completed: u64,
    pub requests_cancelled: u64,
    pub requests_failed: u64,
    pub input_usage_unknown: u64,
}
struct Pending { mode: usize, accepted: u64, done: bool, closed: bool }
#[derive(Default)]
struct State { modes: [Counters; 2], pending: HashMap<u64, Pending> }
pub struct Telemetry { epoch: String, started: f64, state: Mutex<State> }

impl Default for Telemetry {
    fn default() -> Self {
        let time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        Self { epoch: format!("dsfront:{}:{}", std::process::id(), time.as_nanos()),
            started: time.as_secs_f64(), state: Mutex::new(State::default()) }
    }
}
impl Telemetry {
    pub fn begin(&self, id: u64, creative: bool) {
        self.state.lock().unwrap().pending.entry(id).or_insert(Pending {
            mode: usize::from(creative), accepted: 0, done: false, closed: false });
    }
    pub fn abort_submit(&self, id: u64) { self.state.lock().unwrap().pending.remove(&id); }
    pub fn accepted(&self, id: u64, total: usize) {
        let mut s = self.state.lock().unwrap();
        if let Some(p) = s.pending.get_mut(&id) {
            // Idempotent cumulative request position, never retroactively subtract tokens.
            if p.closed || total as u64 <= p.accepted { return; }
            let delta = total as u64 - p.accepted;
            p.accepted = total as u64;
            let mode = p.mode;
            s.modes[mode].output_tokens += delta;
        }
    }
    pub fn engine_done(&self, d: &Done) {
        let mut s = self.state.lock().unwrap();
        if let Some(p) = s.pending.get_mut(&d.req_id) {
            if p.done { return; }
            p.done = true;
            let (mode, closed) = (p.mode, p.closed);
            let c = &mut s.modes[mode];
            if d.reason == DoneReason::Error {
                c.requests_failed += 1;
                c.input_usage_unknown += 1;
            } else {
                c.requests_completed += 1;
                if d.reason == DoneReason::Cancelled { c.requests_cancelled += 1; }
                if d.cached_tokens <= d.prompt_tokens {
                    // Publish prompt/cache as one coherent snapshot, once, even after client abort.
                    c.prompt_tokens += d.prompt_tokens as u64;
                    c.cached_tokens += d.cached_tokens as u64;
                } else { c.input_usage_unknown += 1; }
            }
            if closed { s.pending.remove(&d.req_id); }
        }
    }
    pub fn client_closed(&self, id: u64) {
        let mut s = self.state.lock().unwrap();
        if let Some(p) = s.pending.get_mut(&id) {
            p.closed = true;
            if p.done { s.pending.remove(&id); }
        }
    }
    pub fn snapshot(&self) -> serde_json::Value {
        let s = self.state.lock().unwrap();
        serde_json::json!({"schema":1,"counter_scope":"frontend_process_epoch",
            "runtime_epoch":self.epoch,"process_start_time":self.started,
            "observed_epoch":SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64(),
            "output_scope":"frontend_accepted_token_ids_including_first_and_stop",
            "input_scope":"engine_done_prompt_cache",
            "normal":s.modes[0],"creative":s.modes[1],"pending":s.pending.len()})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn done(id: u64, reason: DoneReason) -> Done {
        Done {req_id:id,reason,prompt_tokens:100,cached_tokens:80,completion_tokens:99,
            queue_us:0,prefill_us:1,decode_us:1,error:String::new()}
    }
    #[test]
    fn live_before_done_and_exact_once_modes() {
        let t=Telemetry::default(); t.begin(1,false); t.begin(2,true);
        t.accepted(1,1);t.accepted(1,9);t.accepted(1,9);t.accepted(1,8);t.accepted(2,4);
        let x=t.snapshot();assert_eq!(x["normal"]["output_tokens"],9);assert_eq!(x["normal"]["prompt_tokens"],0);
        t.engine_done(&done(1,DoneReason::Stop));t.engine_done(&done(1,DoneReason::Stop));
        t.client_closed(1);t.engine_done(&done(2,DoneReason::Length));t.client_closed(2);
        let x=t.snapshot();assert_eq!(x["normal"]["prompt_tokens"],100);assert_eq!(x["normal"]["cached_tokens"],80);
        assert_eq!(x["creative"]["output_tokens"],4);assert_eq!(x["pending"],0);
    }
    #[test]
    fn done_can_arrive_before_parser_and_client_abort_before_done() {
        let t=Telemetry::default();t.begin(1,false);t.engine_done(&done(1,DoneReason::Stop));
        t.accepted(1,8);t.client_closed(1);assert_eq!(t.snapshot()["normal"]["output_tokens"],8);
        t.begin(2,false);t.accepted(2,3);t.client_closed(2);t.accepted(2,12);
        t.engine_done(&done(2,DoneReason::Cancelled));
        let x=t.snapshot();assert_eq!(x["normal"]["output_tokens"],11);assert_eq!(x["normal"]["prompt_tokens"],200);
        assert_eq!(x["normal"]["requests_cancelled"],1);assert_eq!(x["pending"],0);
    }
    #[test]
    fn lost_connection_keeps_observed_output_without_inventing_input() {
        let t=Telemetry::default();t.begin(1,false);t.accepted(1,7);
        t.engine_done(&done(1,DoneReason::Error));t.client_closed(1);
        let x=t.snapshot();assert_eq!(x["normal"]["output_tokens"],7);assert_eq!(x["normal"]["prompt_tokens"],0);
        assert_eq!(x["normal"]["input_usage_unknown"],1);assert_eq!(x["pending"],0);
        assert_ne!(x["runtime_epoch"],Telemetry::default().snapshot()["runtime_epoch"]);
    }
    #[test]
    fn concurrent_snapshots_are_coherent() {
        let t=std::sync::Arc::new(Telemetry::default());let w=t.clone();
        let thread=std::thread::spawn(move || {for id in 0..1000 {w.begin(id,false);w.accepted(id,5);w.engine_done(&done(id,DoneReason::Stop));w.client_closed(id);}});
        for _ in 0..1000 {let x=t.snapshot();let c=&x["normal"];assert_eq!(c["prompt_tokens"].as_u64().unwrap()*80,c["cached_tokens"].as_u64().unwrap()*100);}
        thread.join().unwrap();assert_eq!(t.snapshot()["normal"]["output_tokens"],5000);
    }
}
