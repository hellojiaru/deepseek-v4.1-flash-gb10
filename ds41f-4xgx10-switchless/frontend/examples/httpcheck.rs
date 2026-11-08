use dsfront::server::{prepare, ServerConfig};
use dsfront::encoding::Encoding;
use serde_json::{json,Value};
use std::io::BufRead;
fn main() {
 let args:Vec<String>=std::env::args().collect();
 let f=std::fs::File::open(&args[1]).unwrap();
 let encoding=if args.get(2).map(String::as_str)==Some("v41"){Encoding::V41}else{Encoding::V41_1017};
 let mut bad=0;let mut count=0;
 for l in std::io::BufReader::new(f).lines(){
  let c:Value=serde_json::from_str(&l.unwrap()).unwrap();
  let cfg=ServerConfig{creative:true,vision_prefix_policy:dsfront::server::VisionPrefixPolicy::Legacy,cache_scope:"official1017:assets".into(),vision_socket:Some("unused-cpu-test".into()),vision_revision:Some("official-1017".into()),model_name:"test".into(),api_key:None,max_tokens_cap:393216,default_effort:75,encoding,default_thinking:false,default_temperature:0.0,default_top_p:1.0,loop_guard:None,tokenize_threads:1,dump_dir:None,hold:std::time::Duration::from_secs(1)};
  count+=1;
  let (ok,detail)=match prepare(&cfg,&c["request"]){
   Ok(p)=>{
    let equal=c.get("error").is_none()&&p.prompt==c["prompt"].as_str().unwrap()&&json!(p.images)==c["images"];
    (equal,json!({"prompt_matches":p.prompt==c["prompt"].as_str().unwrap_or(""),"images_match":json!(p.images)==c["images"],"temperature":p.temperature,"top_p":p.top_p,"max_tokens":p.max_tokens,"scope":p.cache_namespace}))
   },
   Err(_)=>(c.get("error").is_some(),json!({"raised":true}))
  };
  if !ok{bad+=1;}
  println!("{}",json!({"name":c["name"],"passed":ok,"detail":detail}));
 }
 println!("{}",json!({"cases":count,"mismatches":bad}));
 if bad>0 {std::process::exit(1)}
}
