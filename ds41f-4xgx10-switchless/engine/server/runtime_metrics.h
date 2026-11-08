// Rank-0 CPU observations only. No CUDA calls, token contents or billing counters.
#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

namespace srv {
inline double metrics_wall() {
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
// Same Linux process identity as the read-only fleet observer: PID:start_ticks.
inline std::string metrics_epoch() {
  std::ifstream f("/proc/self/stat"); std::string line, field; std::getline(f,line);
  const auto end=line.rfind(')');
  if(end==std::string::npos) return {};
  std::istringstream in(line.substr(end+2));
  for(int i=0;i<=19;++i) if(!(in>>field)) return {};
  return std::to_string(getpid())+":"+field;
}
inline std::string metrics_quote(const std::string& value) {
  std::string out="\"";
  for(unsigned char c:value) {
    if(c=='"'||c=='\\') out+='\\',out+=c;
    else if(c<32) {char hex[7];std::snprintf(hex,sizeof hex,"\\u%04x",c);out+=hex;}
    else out+=c;
  }
  return out+'"';
}
#define RUNTIME_METRIC_FIELDS(X) \
  X(sequence) X(requests_admitted) X(prompt_tokens_admitted) X(cached_tokens_admitted) \
  X(prefill_tokens_completed) X(resume_tokens_completed) X(prefill_chunks_completed) \
  X(spec_verify_steps) X(spec_drafts_verified) X(spec_drafts_accepted) \
  X(request_prompt_tokens) X(request_cached_tokens) X(request_prefill_completed) \
  X(active_context_tokens) X(context_capacity_tokens) X(active_sessions) \
  X(cache_capacity_bytes) X(cache_used_bytes) X(cache_active_bytes) X(cache_reclaimable_bytes) \
  X(cache_unattributed_bytes) X(cache_disk_capacity_bytes) X(cache_disk_used_bytes) \
  X(cache_sessions) X(cache_disk_sessions) X(cache_evicted_sessions) X(cache_spill_events) \
  X(max_draft_tokens)
struct RuntimeSnapshot {
#define FIELD(name) uint64_t name=0;
  RUNTIME_METRIC_FIELDS(FIELD)
#undef FIELD
  double updated_epoch=0;
  int phase=0; // 0 idle, 1 preparing/state restore, 2 prefill, 3 decode
  int mode=0;
  bool spec_enabled=false;
};
class RuntimeMetrics {
 public:
  RuntimeMetrics(std::string path,std::string model):path_(std::move(path)),model_(std::move(model)),epoch_(metrics_epoch()) {
    const char* off=std::getenv("DSGB10_RUNTIME_METRICS");
    enabled_=!epoch_.empty()&&(!off||std::string(off)!="0");
    if(enabled_) worker_=std::thread([this]{loop();});
  }
  ~RuntimeMetrics() {
    {std::lock_guard<std::mutex> lock(mutex_);stop_=true;}
    wake_.notify_all();if(worker_.joinable())worker_.join();
  }
  template<class F> void update(F f) {
    if(!enabled_)return;
    std::lock_guard<std::mutex> lock(mutex_);f(data_);++data_.sequence;data_.updated_epoch=metrics_wall();
  }
  RuntimeSnapshot snapshot() {std::lock_guard<std::mutex> lock(mutex_);return data_;}
  void verify(uint64_t proposed,uint64_t accepted) {
    update([&](auto& d){++d.spec_verify_steps;d.spec_drafts_verified+=proposed;d.spec_drafts_accepted+=accepted;});
  }
  std::string json(const RuntimeSnapshot& d) const {
    const char* phases[]={"idle","preparing","prefill","decode"};
    std::ostringstream o;o<<std::setprecision(17);
    o<<"{\"schema\":1,\"source\":\"rank0_engine_progress\",\"runtime_epoch\":"<<metrics_quote(epoch_)
     <<",\"model_name\":"<<metrics_quote(model_)<<",\"observed_epoch\":"<<metrics_wall()
     <<",\"updated_epoch\":"<<d.updated_epoch<<",\"phase\":"<<metrics_quote(phases[std::clamp(d.phase,0,3)])
     <<",\"mode\":"<<metrics_quote(d.mode?"creative":"normal")<<",\"spec_enabled\":"<<(d.spec_enabled?"true":"false")
     <<",\"prefill_scope\":\"completed_new_prompt_rows_excluding_reuse_and_resume\""
     <<",\"spec_scope\":\"verified_draft_rows_before_stop_truncation_excluding_seed_and_tail\""
     <<",\"cache_scope\":\"rank0_session_store_excludes_working_buffers\"";
#define EMIT(name) o<<",\"" #name "\":"<<d.name;
    RUNTIME_METRIC_FIELDS(EMIT)
#undef EMIT
    return o.str()+"}";
  }
 private:
  void publish(const RuntimeSnapshot& d) {
    // No file I/O while holding the model-observation mutex. Atomic replacement
    // lets readers see the previous complete document during a write.
    const std::string tmp=path_+".tmp."+std::to_string(getpid()),body=json(d);
    int fd=open(tmp.c_str(),O_WRONLY|O_CREAT|O_TRUNC|O_CLOEXEC,0600);
    bool ok=fd>=0;
    size_t at=0;
    while(ok&&at<body.size()) {ssize_t n=write(fd,body.data()+at,body.size()-at);if(n<=0)ok=false;else at+=n;}
    if(fd>=0&&close(fd))ok=false;
    if(ok)ok=rename(tmp.c_str(),path_.c_str())==0;
    if(!ok) {unlink(tmp.c_str());if(!failed_)std::fprintf(stderr,"runtime metrics: snapshot unavailable (inference continues)\n");}
    failed_=!ok;
  }
  void loop() {
    for(;;) {
      RuntimeSnapshot s;
      {std::unique_lock<std::mutex> lock(mutex_);if(wake_.wait_for(lock,std::chrono::milliseconds(500),[&]{return stop_;}))break;s=data_;}
      publish(s);
    }
  }
  std::string path_,model_,epoch_;
  std::mutex mutex_;std::condition_variable wake_;RuntimeSnapshot data_;
  bool enabled_=false,stop_=false,failed_=false;std::thread worker_;
};
// Only the rank-0 server thread binds this pointer. Worker ranks and benchmarks
// keep it null; they never multiply the counters by the tensor-parallel size.
inline thread_local RuntimeMetrics* g_runtime_metrics=nullptr;
struct RuntimeMetricsScope {
  explicit RuntimeMetricsScope(RuntimeMetrics& m){g_runtime_metrics=&m;}
  ~RuntimeMetricsScope(){g_runtime_metrics=nullptr;}
};
} // namespace srv
