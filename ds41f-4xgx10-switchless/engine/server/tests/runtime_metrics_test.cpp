#include "server/runtime_metrics.h"
#include <cassert>
#include <atomic>
#include <filesystem>
#include <iostream>
#include <sys/stat.h>

int main(int argc,char** argv) {
  assert(argc==2);const std::string path=argv[1];
  double started=srv::metrics_wall();
  {
    srv::RuntimeMetrics m(path,"test-model");
    srv::RuntimeMetricsScope scope(m);
    m.update([](auto& d){d.request_prompt_tokens=10000;d.request_cached_tokens=8000;d.phase=2;});
    m.update([](auto& d){d.prefill_tokens_completed+=1000;d.request_prefill_completed+=1000;});
    // A cancellation after one chunk counts only that finished chunk.
    assert(m.snapshot().request_prefill_completed==1000);
    m.verify(5,3);m.verify(2,0);m.verify(0,0);
    auto a=m.snapshot();assert(a.spec_verify_steps==3&&a.spec_drafts_verified==7&&a.spec_drafts_accepted==3);
    std::atomic<bool> stop{false};
    std::thread reader([&]{while(!stop){auto d=m.snapshot();assert(d.spec_drafts_accepted<=d.spec_drafts_verified);}});
    double t=srv::metrics_wall();
    for(int i=0;i<100000;++i)m.verify(5,3);
    double elapsed=srv::metrics_wall()-t;stop=true;reader.join();
    a=m.snapshot();assert(a.spec_verify_steps==100003&&a.spec_drafts_verified==500007&&a.spec_drafts_accepted==300003);
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    std::ifstream f(path);std::string body((std::istreambuf_iterator<char>(f)),{});
    assert(body.find("\"spec_drafts_verified\":500007")!=std::string::npos);
    assert(body.find("\"request_prefill_completed\":1000")!=std::string::npos);
    struct stat st{};assert(stat(path.c_str(),&st)==0&&(st.st_mode&0777)==0600);
    std::cout<<"{\"passed\":true,\"updates\":100000,\"update_seconds\":"<<elapsed<<",\"atomic_snapshot\":true,\"private_mode\":true,\"runtime_epoch\":"<<srv::metrics_quote(srv::metrics_epoch())<<"}\n";
  }
  assert(srv::g_runtime_metrics==nullptr);
  // A read-only/unavailable output path must not abort inference.
  {srv::RuntimeMetrics m("/proc/no-such-metrics-directory/file","test");m.verify(1,1);std::this_thread::sleep_for(std::chrono::milliseconds(600));}
  assert(srv::metrics_wall()-started<10);
}
