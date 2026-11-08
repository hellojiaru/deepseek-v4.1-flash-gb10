//! CPU integration: real tokenizer/session planner/cache and Unix wire, synthetic
//! embeddings only. This does NOT qualify GPU numerical or visual correctness.
use dsfront::{vision::{self,FeatureCache,IMAGE_ID},tokenizer::Tokenizer,session::SessionTable};
use serde_json::{json,Value};
use std::sync::{Arc,Mutex};
use tokio::{net::UnixListener,io::{AsyncReadExt,AsyncWriteExt}};

#[tokio::test]
#[ignore="requires DSGB10_TEST_TOKENIZER and >=700 MiB disk spool; isolated CPU only"]
async fn fifty_eight_distinct_full_resolution_images_are_batched_spooled_and_reused() {
    let tok=Tokenizer::from_file(&std::env::var("DSGB10_TEST_TOKENIZER").unwrap()).unwrap();
    let path=std::env::temp_dir().join(format!("vision-history58-cpu-{}.sock",std::process::id()));
    let listener=UnixListener::bind(&path).unwrap();
    let counts=Arc::new(Mutex::new(Vec::<usize>::new()));let seen=counts.clone();
    let task=tokio::spawn(async move {
        loop {
            let (mut stream,_)=listener.accept().await.unwrap();
            let n=stream.read_u32_le().await.unwrap();let mut b=vec![0;n as usize];stream.read_exact(&mut b).await.unwrap();
            let body:Value=serde_json::from_slice(&b).unwrap();
            let images=body.get("images").and_then(Value::as_array);
            if let Some(images)=images { assert!(images.len()<=8);seen.lock().unwrap().push(images.len()); }
            let meta:Vec<_>=images.into_iter().flatten().map(|r| {
                let id=r["data"].as_str().unwrap().parse::<u64>().unwrap();
                json!({"rows":1024,"dim":5120,"digest":format!("{id:064x}")})
            }).collect();
            let header=json!({"ok":true,"revision":"test-rev","identity":"11".repeat(32),"images":meta});
            let b=serde_json::to_vec(&header).unwrap();stream.write_u32_le(b.len() as u32).await.unwrap();stream.write_all(&b).await.unwrap();
            for _ in &meta { stream.write_all(&vec![0;1024*5120*2]).await.unwrap(); }
        }
    });
    let mut cache=FeatureCache::default();
    let records:Vec<_>=(0..58).map(|i|json!({"data":i.to_string()})).collect();
    let raw=(0..58).map(|i|format!("Turn {i}: <｜deepseek_image｜>\n")).collect::<String>();
    let a=vision::materialize_cached(&mut cache,path.to_str().unwrap(),"test-rev",&records,&raw,"normal",&tok,1).await.unwrap();
    assert_eq!(a.images.len(),58);
    assert_eq!(a.images.iter().map(|i|i.feature.len()).sum::<usize>(),580<<20);
    assert_eq!(*counts.lock().unwrap(),vec![8,8,8,8,8,8,8,2]);
    let ids=tok.encode(&a.prompt);
    for (index,image) in a.images.iter().enumerate() {
        assert_eq!(std::fs::metadata(image.feature.path()).unwrap().len(),10<<20);
        assert_eq!(image.digest[31],index as u8);
        assert!(ids[image.start as usize..(image.start+image.rows) as usize].iter().all(|&id|id==IMAGE_ID));
    }
    let b=vision::materialize_cached(&mut cache,path.to_str().unwrap(),"test-rev",&records,&raw,"normal",&tok,1).await.unwrap();
    assert_eq!(counts.lock().unwrap().len(),8,"unchanged history must not re-encode");
    assert!(a.images.iter().zip(&b.images).all(|(a,b)|a.feature.shares_owner(&b.feature)));
    let mut appended=records.clone();appended.extend((58..62).map(|i|json!({"data":i.to_string()})));
    let more=format!("{raw}{}", "<｜deepseek_image｜>\n".repeat(4));
    let c=vision::materialize_cached(&mut cache,path.to_str().unwrap(),"test-rev",&appended,&more,"normal",&tok,1).await.unwrap();
    assert_eq!(counts.lock().unwrap().last().copied(),Some(4));
    assert_eq!(c.images.len(),62);
    let paths:Vec<_>=c.images.iter().map(|i|i.feature.path().to_owned()).collect();
    drop(cache);drop(a);drop(b);
    assert!(paths.iter().all(|p|std::path::Path::new(p).exists()),"inflight owners outlive cache");
    drop(c);
    assert!(paths.iter().all(|p|!std::path::Path::new(p).exists()));
    task.abort();let _=task.await;std::fs::remove_file(path).unwrap();
}

#[tokio::test]
#[ignore="requires DSGB10_TEST_TOKENIZER; isolated CPU only"]
async fn long_text_first_append_replace_remove_modes_and_encoder_generation() {
    let tok=Tokenizer::from_file(&std::env::var("DSGB10_TEST_TOKENIZER").unwrap()).unwrap();
    let path=std::env::temp_dir().join(format!("vision-cache-cpu-{}.sock",std::process::id()));
    let listener=UnixListener::bind(&path).unwrap();
    let counts=Arc::new(Mutex::new(Vec::<usize>::new()));let seen=counts.clone();
    let identity=Arc::new(Mutex::new("11".repeat(32)));let server_identity=identity.clone();
    let task=tokio::spawn(async move {
        loop {
            let (mut stream,_)=listener.accept().await.unwrap();
            let n=stream.read_u32_le().await.unwrap();let mut b=vec![0;n as usize];stream.read_exact(&mut b).await.unwrap();
            let body:Value=serde_json::from_slice(&b).unwrap();let id=server_identity.lock().unwrap().clone();
            let images=body.get("images").and_then(Value::as_array);
            if let Some(images)=images {seen.lock().unwrap().push(images.len());}
            let meta:Vec<_>=images.into_iter().flatten().map(|r| {
                let n=r["data"].as_str().unwrap().parse::<u64>().unwrap();json!({"rows":32,"dim":5120,"digest":format!("{n:064x}")})
            }).collect();
            let header=json!({"ok":true,"revision":"test-rev","identity":id,"images":meta});
            let b=serde_json::to_vec(&header).unwrap();stream.write_u32_le(b.len() as u32).await.unwrap();stream.write_all(&b).await.unwrap();
            for _ in &meta {stream.write_all(&vec![0;32*5120*2]).await.unwrap();}
        }
    });
    // Strings are immutable inline data; numeric IDs live in an extra test field.
    // Server accepts strings to exercise the same cache-key rule as production data.
    let mut cache=FeatureCache::default();
    for mode in ["normal","creative"] {
        let namespace=format!("official1017:{mode}:full-v1");let mut table=SessionTable::new(32,1<<20);
        let raw=format!("<｜begin▁of▁sentence｜><｜System｜>Keep all prior facts and inspect the exact image pixels.<｜User｜>{}<｜Assistant｜>Ready.<｜end▁of▁sentence｜>","stable long history with distinct numbered facts.\n".repeat(12000));
        let a=vision::materialize_cached(&mut cache,path.to_str().unwrap(),"test-rev",&[],&raw,&namespace,&tok,2).await.unwrap();
        let (p,_)=table.plan_images(&a.prompt,&a.scope,&a.cache_images,&tok,2);assert!(p.ids.len()>60000);
        table.finish(p.session_id,&a.prompt,&p.ids,&[],&tok);let old=p.ids.len(); let mut previous=old;
        let mut prompt=raw;let mut records=Vec::new();
        for n in 1..=4 {
            prompt.push_str("<｜User｜>Picture: <｜deepseek_image｜><｜Assistant｜>Seen.<｜end▁of▁sentence｜>");
            records.push(json!({"data":n.to_string()}));
            let expanded=vision::materialize_cached(&mut cache,path.to_str().unwrap(),"test-rev",&records,&prompt,&namespace,&tok,2).await.unwrap();
            let (next,_)=table.plan_images(&expanded.prompt,&expanded.scope,&expanded.cache_images,&tok,2);
            assert!(next.reused>=previous,"first/append image discarded the old prefix"); previous=next.ids.len();
            assert_eq!(next.ids,tok.encode(&expanded.prompt));
            for im in &expanded.images {assert!(next.ids[im.start as usize..(im.start+im.rows) as usize].iter().all(|&id|id==IMAGE_ID));}
            table.finish(next.session_id,&expanded.prompt,&next.ids,&[],&tok);
            let before=counts.lock().unwrap().len();
            let warm=vision::materialize_cached(&mut cache,path.to_str().unwrap(),"test-rev",&records,&prompt,&namespace,&tok,2).await.unwrap();
            assert_eq!(before,counts.lock().unwrap().len(),"historical bytes resent to encoder");
            assert!(warm.images[0].feature.shares_owner(&expanded.images[0].feature));
            if n==4 {
                let first=expanded.images[0].start as usize;records[0]=json!({"data":"99"});
                let replaced=vision::materialize_cached(&mut cache,path.to_str().unwrap(),"test-rev",&records,&prompt,&namespace,&tok,2).await.unwrap();
                let (rp,_)=table.plan_images(&replaced.prompt,&replaced.scope,&replaced.cache_images,&tok,2);
                assert!(rp.reused<=first && rp.reused>=old);assert_ne!(replaced.images[0].digest,expanded.images[0].digest);
                assert_eq!(rp.ids,tok.encode(&replaced.prompt));
                table.finish(rp.session_id,&replaced.prompt,&rp.ids,&[],&tok);
                let (isolated,_)=table.plan_images(&replaced.prompt,&format!("{namespace}:other-mode"),&replaced.cache_images,&tok,2);assert_eq!(isolated.reused,0);
                let cleared=vision::materialize_cached(&mut cache,path.to_str().unwrap(),"test-rev",&[],&a.prompt,&namespace,&tok,2).await.unwrap();
                let (removed,_)=table.plan_images(&cleared.prompt,&cleared.scope,&cleared.cache_images,&tok,2);assert!(removed.reused<=old);
            }
        }
    }
    let before=counts.lock().unwrap().len();*identity.lock().unwrap()="22".repeat(32);
    let raw="<｜begin▁of▁sentence｜><｜User｜><｜deepseek_image｜><｜Assistant｜>";
    vision::materialize_cached(&mut cache,path.to_str().unwrap(),"test-rev",&[json!({"data":"1"})],raw,"normal",&tok,1).await.unwrap();
    assert_eq!(counts.lock().unwrap().len(),before+1);
    task.abort();std::fs::remove_file(path).unwrap();
}
