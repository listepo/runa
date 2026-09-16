//! `runa fit --recommend` (P8.4): rank a curated model list for this machine.
//!
//! Each catalog entry is fitted in parallel, from its GGUF header (remote,
//! cached after the first fetch) or, offline, from the sizes stored in the
//! catalog. NO FIT drops out; the rest rank by usable speed (≥ 5 tok/s)
//! first, then quality tier, then predicted decode tok/s.

use serde::Deserialize;

use crate::descriptor::Descriptor;
use crate::gguf::Reader;
use crate::remote::{Fetcher, parse_model_ref};
use crate::speed::{HwSpec, estimate_speed_hybrid};
use crate::verdict::{FitConfig, FitReport, Verdict, check_fit};

const CATALOG: &str = include_str!("catalog.toml");

/// Below this a model fits but is slow to use (the `DecodeSlow` bar).
pub const USABLE_TOKS_PER_SEC: f64 = 5.0;

/// Values `--use` accepts.
pub const USES: [&str; 4] = ["chat", "code", "vision", "reasoning"];

/// One model in the embedded catalog.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct CatalogEntry {
    pub name: String,
    /// Exact `hf:<repo>:<file.gguf>` to pull.
    #[serde(rename = "ref")]
    pub model: String,
    pub uses: Vec<String>,
    /// 1 (small) … 4 (best in the catalog).
    pub tier: u8,
    /// File size in bytes.
    pub size: u64,
    /// Approximate bytes read per token (MoE); `None` = `size`.
    #[serde(default)]
    pub active: Option<u64>,
    /// Projector file size (vision models), 0 when there is none.
    #[serde(default)]
    pub mmproj: u64,
}

#[derive(Deserialize)]
struct Catalog {
    model: Vec<CatalogEntry>,
}

/// The embedded catalog (`catalog.toml`).
pub fn catalog() -> Vec<CatalogEntry> {
    toml::from_str::<Catalog>(CATALOG)
        .expect("embedded catalog.toml parses (see catalog_is_sane)")
        .model
}

/// One fitted catalog entry.
#[derive(Debug, Clone, PartialEq)]
pub struct Pick {
    pub entry: CatalogEntry,
    pub verdict: Verdict,
    pub decode_toks_per_sec: f64,
    /// From catalog sizes only (`--offline`), not the header.
    pub estimated: bool,
}

impl Pick {
    fn usable(&self) -> bool {
        self.decode_toks_per_sec >= USABLE_TOKS_PER_SEC
    }
}

/// Probe every entry for `use_` in parallel and rank what fits. Returns the
/// top `top` picks and `(name, error)` for each probe that failed.
pub fn recommend(
    entries: &[CatalogEntry],
    use_: Option<&str>,
    top: usize,
    probe: impl Fn(&CatalogEntry) -> Result<Pick, String> + Sync,
) -> (Vec<Pick>, Vec<(String, String)>) {
    let wanted: Vec<&CatalogEntry> = entries
        .iter()
        .filter(|e| use_.is_none_or(|u| e.uses.iter().any(|x| x == u)))
        .collect();
    let probe = &probe;
    let results: Vec<Result<Pick, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = wanted.iter().map(|&e| s.spawn(move || probe(e))).collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err("probe panicked".into())))
            .collect()
    });
    let mut picks = Vec::new();
    let mut errors = Vec::new();
    for (e, r) in wanted.iter().zip(results) {
        match r {
            Ok(p) if p.verdict != Verdict::NoFit => picks.push(p),
            Ok(_) => {}
            Err(msg) => errors.push((e.name.clone(), msg)),
        }
    }
    picks.sort_by(|a, b| {
        b.usable()
            .cmp(&a.usable())
            .then(b.entry.tier.cmp(&a.entry.tier))
            .then(b.decode_toks_per_sec.total_cmp(&a.decode_toks_per_sec))
    });
    picks.truncate(top);
    (picks, errors)
}

/// Decode tok/s for the placement the verdict chose.
pub fn decode_for(desc: &Descriptor, report: &FitReport, config: &FitConfig) -> f64 {
    let gpu = report.speed_gpu.as_ref().map(|s| s.decode_toks_per_sec);
    let cpu = report.speed_cpu.as_ref().map(|s| s.decode_toks_per_sec);
    let speed = match report.verdict {
        Verdict::Gpu => gpu.or(cpu),
        Verdict::Cpu | Verdict::NoFit => cpu,
        Verdict::Hybrid { .. } => {
            // ponytail: the GPU share is by all weight bytes, not the active
            // ones, so MoE with experts on CPU reads a little slow here.
            let plan = &report.plan;
            let total = plan.gpu_weight_bytes + plan.cpu_weight_bytes;
            let frac = plan.gpu_weight_bytes as f64 / total.max(1) as f64;
            config
                .gpu_hw
                .as_ref()
                .map(|hw| {
                    estimate_speed_hybrid(
                        desc,
                        &report.kv_estimate,
                        config.planner.ctx_len,
                        1024,
                        hw,
                        &config.cpu_hw,
                        frac,
                    )
                    .decode_toks_per_sec
                })
                .or(cpu)
        }
    };
    speed.unwrap_or(0.0)
}

/// Fit one entry from its GGUF header (the fetcher caches it on disk).
pub fn probe_remote(
    fetcher: &Fetcher,
    entry: &CatalogEntry,
    config: &FitConfig,
) -> Result<Pick, String> {
    let src = parse_model_ref(&entry.model).map_err(|e| e.to_string())?;
    let header = fetcher.fetch_header(&src).map_err(|e| e.to_string())?;
    let reader = Reader::parse(&header.bytes).map_err(|e| e.to_string())?;
    let desc = Descriptor::from_reader(&reader).map_err(|e| e.to_string())?;
    let report = check_fit(&desc, config);
    Ok(Pick {
        entry: entry.clone(),
        decode_toks_per_sec: decode_for(&desc, &report, config),
        verdict: report.verdict,
        estimated: false,
    })
}

/// Fit one entry from catalog sizes: weights (+ projector) against VRAM
/// minus its margin, else RAM.
// ponytail: KV cache and compute buffers are not counted offline (the header
// knows them); a long context on a big model can fit here and not online.
pub fn probe_offline(entry: &CatalogEntry, config: &FitConfig) -> Pick {
    let p = &config.planner;
    let need = entry.size + p.mmproj_bytes;
    let active = entry.active.unwrap_or(entry.size).max(1) as f64;
    let speed = |hw: &HwSpec| hw.efficiency * hw.bandwidth_gbps * 1e9 / active;
    let (verdict, decode) = match &config.gpu_hw {
        Some(hw) if need + p.vram_margin <= p.vram_bytes => (Verdict::Gpu, speed(hw)),
        _ if need <= p.ram_bytes => (Verdict::Cpu, speed(&config.cpu_hw)),
        _ => (Verdict::NoFit, 0.0),
    };
    Pick {
        entry: entry.clone(),
        verdict,
        decode_toks_per_sec: decode,
        estimated: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::PlannerConfig;

    fn entry(name: &str, tier: u8, uses: &[&str]) -> CatalogEntry {
        CatalogEntry {
            name: name.into(),
            model: format!("hf:x/{name}:{name}.gguf"),
            uses: uses.iter().map(|u| u.to_string()).collect(),
            tier,
            size: 4 << 30,
            active: None,
            mmproj: 0,
        }
    }

    #[test]
    fn catalog_is_sane() {
        let c = catalog();
        assert!(c.len() >= 10);
        for e in &c {
            assert!(
                e.model.starts_with("hf:") && e.model.ends_with(".gguf"),
                "{e:?}"
            );
            assert!((1..=4).contains(&e.tier), "{e:?}");
            assert!(!e.uses.is_empty() && e.uses.iter().all(|u| USES.contains(&u.as_str())));
            assert!(e.size > 0 && e.active.is_none_or(|a| a < e.size), "{e:?}");
            assert_eq!(c.iter().filter(|x| x.name == e.name).count(), 1, "{e:?}");
        }
        for u in USES {
            assert!(c.iter().any(|e| e.uses.iter().any(|x| x == u)), "{u}");
        }
    }

    #[test]
    fn ranks_usable_then_tier_then_speed() {
        let entries = [
            entry("slow-best", 4, &["chat"]),
            entry("mid", 3, &["chat"]),
            entry("mid-fast", 3, &["chat", "code"]),
            entry("too-big", 4, &["chat"]),
            entry("broken", 4, &["chat"]),
            entry("coder", 2, &["code"]),
        ];
        let fake = |e: &CatalogEntry| {
            let (verdict, tps) = match e.name.as_str() {
                "slow-best" => (Verdict::Cpu, 3.0),
                "mid" => (Verdict::Gpu, 20.0),
                "mid-fast" => (Verdict::Gpu, 40.0),
                "too-big" => (Verdict::NoFit, 0.0),
                "broken" => return Err("http 404".to_string()),
                _ => (Verdict::Gpu, 90.0),
            };
            Ok(Pick {
                entry: e.clone(),
                verdict,
                decode_toks_per_sec: tps,
                estimated: false,
            })
        };
        let names = |picks: &[Pick]| {
            picks
                .iter()
                .map(|p| p.entry.name.clone())
                .collect::<Vec<_>>()
        };

        let (picks, errors) = recommend(&entries, Some("chat"), 10, fake);
        assert_eq!(names(&picks), ["mid-fast", "mid", "slow-best"]);
        assert_eq!(errors, [("broken".to_string(), "http 404".to_string())]);

        let (picks, _) = recommend(&entries, None, 2, fake);
        assert_eq!(names(&picks), ["mid-fast", "mid"]);

        let (picks, errors) = recommend(&entries, Some("code"), 10, fake);
        assert_eq!(names(&picks), ["mid-fast", "coder"]);
        assert!(errors.is_empty());
    }

    #[test]
    fn offline_fits_on_sizes() {
        let config = |vram_gib: u64, ram_gib: u64| FitConfig {
            planner: PlannerConfig {
                vram_bytes: vram_gib << 30,
                ram_bytes: ram_gib << 30,
                ..PlannerConfig::default()
            },
            gpu_hw: Some(HwSpec::metal()),
            ..FitConfig::default()
        };
        let dense = entry("dense", 3, &["chat"]);
        let moe = CatalogEntry {
            active: Some(1 << 30),
            ..dense.clone()
        };
        assert_eq!(probe_offline(&dense, &config(8, 16)).verdict, Verdict::Gpu);
        assert_eq!(probe_offline(&dense, &config(4, 16)).verdict, Verdict::Cpu);
        assert_eq!(probe_offline(&dense, &config(4, 2)).verdict, Verdict::NoFit);
        let d = probe_offline(&dense, &config(8, 16)).decode_toks_per_sec;
        let m = probe_offline(&moe, &config(8, 16)).decode_toks_per_sec;
        assert!(m > 3.0 * d, "{m} vs {d}");
    }
}
