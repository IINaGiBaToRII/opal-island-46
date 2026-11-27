
use engine_core::{search_shared, GameConfig, SearchLimits, SearchResultOut, TtHandle, VariantId};
use serde_json::{json, Value};
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

enum Ev {
    Line(Option<String>),
    Done(Result<SearchResultOut, String>, TtHandle),
}

fn out(v: Value) {
    println!("{v}");
}

pub fn run() -> Result<(), String> {
    let (tx, rx) = mpsc::channel::<Ev>();
    let txr = tx.clone();
    std::thread::spawn(move || {
        for l in std::io::stdin().lock().lines() {
            match l {
                Ok(l) => {
                    if txr.send(Ev::Line(Some(l))).is_err() {
                        return;
                    }
                }
                Err(_) => break,
            }
        }
        let _ = txr.send(Ev::Line(None));
    });
    let mut cfg = GameConfig { variant: VariantId::Russian, forced_capture: true };
    let mut limits_base = SearchLimits::default();
    let mut dfen: Option<String> = None;
    let mut hist: Vec<String> = vec![];
    let mut handle = Some(TtHandle::new());
    let stop = Arc::new(AtomicBool::new(false));
    let mut searching = false;
    let mut quit_after = false;
    loop {
        let ev = match rx.recv() {
            Ok(e) => e,
            Err(_) => break,
        };
        match ev {
            Ev::Line(None) => {
                stop.store(true, Ordering::Relaxed);
                quit_after = true;
                if !searching {
                    break;
                }
            }
            Ev::Done(r, h) => {
                handle = Some(h);
                searching = false;
                match r {
                    Ok(r) => out(json!({"type":"bestmove","move":r.best_move,"score":r.score,"depth":r.depth,"nodes":r.nodes,"time_ms":r.time_ms,"pv":r.pv,"eval_id":r.eval_id})),
                    Err(e) => out(json!({"type":"error","msg":e})),
                }
                if quit_after {
                    break;
                }
            }
            Ev::Line(Some(l)) => {
                let l = l.trim();
                if l.is_empty() {
                    continue;
                }
                let v: Value = match serde_json::from_str(l) {
                    Ok(v) => v,
                    Err(e) => {
                        out(json!({"type":"error","msg":format!("bad json: {e}")}));
                        continue;
                    }
                };
                let cmd = v.get("cmd").and_then(|c| c.as_str()).unwrap_or("");
                if searching && cmd != "stop" && cmd != "quit" && cmd != "isready" {
                    out(json!({"type":"error","msg":"search in progress"}));
                    continue;
                }
                match cmd {
                    "setoption" => {
                        if let Some(n) = v.get("variant").and_then(|x| x.as_str()) {
                            match VariantId::parse(n) {
                                Some(id) => cfg.variant = id,
                                None => out(json!({"type":"error","msg":format!("unknown variant {n}")})),
                            }
                        }
                        if let Some(f) = v.get("forced").and_then(|x| x.as_bool()) {
                            cfg.forced_capture = f;
                        }
                        if let Some(h) = v.get("hash_mb").and_then(|x| x.as_u64()) {
                            limits_base.hash_mb = h as usize;
                        }
                        if let Some(t) = v.get("threads").and_then(|x| x.as_u64()) {
                            limits_base.threads = t as usize;
                        }
                        if let Some(o) = v.get("opts").and_then(|x| x.as_str()) {
                            limits_base.opts = o.to_string();
                        }
                    }
                    "newgame" => handle = Some(TtHandle::new()),
                    "isready" => out(json!({"type":"readyok"})),
                    "position" => {
                        dfen = v.get("dfen").and_then(|x| x.as_str()).map(|s| s.to_string());
                        hist = v
                            .get("history")
                            .and_then(|x| x.as_array())
                            .map(|a| a.iter().filter_map(|s| s.as_str().map(|s| s.to_string())).collect())
                            .unwrap_or_default();
                    }
                    "go" => {
                        let Some(fen) = dfen.clone() else {
                            out(json!({"type":"error","msg":"no position"}));
                            continue;
                        };
                        let mut lim = limits_base.clone();
                        lim.depth = v.get("depth").and_then(|x| x.as_u64()).map(|x| x as u32);
                        lim.time_ms = v.get("time_ms").and_then(|x| x.as_u64());
                        lim.nodes = v.get("nodes").and_then(|x| x.as_u64());
                        if lim.depth.is_none() && lim.time_ms.is_none() && lim.nodes.is_none() {
                            lim.time_ms = Some(1000);
                        }
                        stop.store(false, Ordering::Relaxed);
                        searching = true;
                        let mut h = handle.take().unwrap_or_default();
                        let (stp, txd, hs) = (stop.clone(), tx.clone(), hist.clone());
                        let spawned = std::thread::Builder::new().stack_size(256 << 20).spawn(move || {
                            let r = search_shared(&cfg, &fen, &hs, &lim, &stp, &mut |i| {
                                out(json!({"type":"info","depth":i.depth,"score":i.score,"nodes":i.nodes,"nps":i.nps,"time_ms":i.time_ms,"pv":i.pv}))
                            }, &mut h);
                            let _ = txd.send(Ev::Done(r, h));
                        });
                        if let Err(e) = spawned {
                            searching = false;
                            out(json!({"type":"error","msg":e.to_string()}));
                        }
                    }
                    "stop" => stop.store(true, Ordering::Relaxed),
                    "quit" => {
                        stop.store(true, Ordering::Relaxed);
                        quit_after = true;
                        if !searching {
                            break;
                        }
                    }
                    _ => out(json!({"type":"error","msg":format!("unknown cmd '{cmd}'")})),
                }
            }
        }
    }
    Ok(())
}
