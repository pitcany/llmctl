//! Model acquisition (M4c item 1), the pure part: where weights go (a
//! declared store), what a source string means, how much a download needs
//! and whether the disk has it.
//!
//! Conversion stays with hf2local (owner decision 7): its output
//! directories are stores of kind `directory`, recorded, not rebuilt.

use serde::{Deserialize, Serialize};

use crate::identity::WorkloadName;

/// What kind of place a store is, which decides how weights get into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoreKind {
    /// A Hugging Face cache (`hub/models--org--name/…`).
    HfCache,
    /// Plain model directories, one per model (hf2local output, `~/models`).
    Directory,
    /// An Ollama server's model store; filled through that server.
    Ollama,
}

/// A place model weights live, declared in the inventory. Stores are
/// separate records from engines: removing an engine never removes
/// weights (design decision 5).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStore {
    pub name: String,
    pub kind: StoreKind,
    /// Absolute path of the store's root.
    pub path: String,
    /// Free space that must remain on the store's filesystem after a
    /// download (GiB). mbp2 sat at 93 % full when this was written.
    #[serde(default = "default_reserve_gib")]
    pub reserve_gib: u64,
    /// The workload whose journal an acquisition into this store is
    /// recorded under; for `ollama` also the server that pulls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workload: Option<WorkloadName>,
}

fn default_reserve_gib() -> u64 {
    20
}

/// Where weights come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// `hf:org/name[@revision]`
    Hf { repo: String, revision: Option<String> },
    /// `ollama:tag`
    Ollama { tag: String },
}

impl Source {
    /// Parse `hf:org/name[@rev]` or `ollama:tag`. Refuses anything that
    /// could be read as an option or a path by the tool that receives it.
    pub fn parse(text: &str) -> Result<Source, String> {
        let plain = |s: &str, what: &str| -> Result<(), String> {
            let ok = !s.is_empty()
                && !s.starts_with('-')
                && !s.contains("..")
                && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/' | ':'));
            if ok { Ok(()) } else { Err(format!("{what} {s:?} is not a plain model name")) }
        };
        match text.split_once(':') {
            Some(("hf", rest)) => {
                let (repo, revision) = match rest.split_once('@') {
                    Some((r, rev)) => (r, Some(rev)),
                    None => (rest, None),
                };
                plain(repo, "repository")?;
                if repo.matches('/').count() != 1 || repo.contains(':') {
                    return Err(format!("{repo:?} is not an `org/name` repository"));
                }
                if let Some(rev) = revision {
                    plain(rev, "revision")?;
                }
                Ok(Source::Hf { repo: repo.to_string(), revision: revision.map(str::to_owned) })
            }
            Some(("ollama", tag)) => {
                plain(tag, "tag")?;
                Ok(Source::Ollama { tag: tag.to_string() })
            }
            _ => Err(format!("{text:?} is not a source: use hf:org/name[@revision] or ollama:tag")),
        }
    }

    /// Whether a store of `kind` can receive this source.
    pub fn fits_store(&self, kind: StoreKind) -> bool {
        matches!(
            (self, kind),
            (Source::Hf { .. }, StoreKind::HfCache | StoreKind::Directory)
                | (Source::Ollama { .. }, StoreKind::Ollama)
        )
    }
}

impl std::fmt::Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Source::Hf { repo, revision: Some(r) } => write!(f, "hf:{repo}@{r}"),
            Source::Hf { repo, revision: None } => write!(f, "hf:{repo}"),
            Source::Ollama { tag } => write!(f, "ollama:{tag}"),
        }
    }
}

/// Total bytes from `hf download … --dry-run --format json`
/// (`[{"file": "model.safetensors", "size": "453.9K"}, …]`, decimal
/// units). `None` when the listing cannot be understood: an unknown size
/// is not zero.
pub fn hf_dry_run_bytes(json: &str) -> Option<u64> {
    let rows: Vec<serde_json::Value> = serde_json::from_str(json.trim()).ok()?;
    let mut total = 0f64;
    for row in &rows {
        total += human_bytes(row.get("size")?.as_str()?)?;
    }
    (!rows.is_empty()).then_some(total.ceil() as u64)
}

/// `"453.9K"` → bytes (K/M/G/T are powers of 1000, as the hub prints them).
/// `"-"` is a file that is already in the store: nothing left to fetch.
fn human_bytes(text: &str) -> Option<f64> {
    let text = text.trim();
    if text == "-" {
        return Some(0.0);
    }
    let (number, factor) = match text.chars().last()? {
        'K' => (&text[..text.len() - 1], 1e3),
        'M' => (&text[..text.len() - 1], 1e6),
        'G' => (&text[..text.len() - 1], 1e9),
        'T' => (&text[..text.len() - 1], 1e12),
        _ => (text, 1.0),
    };
    let n: f64 = number.trim().parse().ok()?;
    (n >= 0.0).then_some(n * factor)
}

pub const GIB: u64 = 1024 * 1024 * 1024;

/// The disk check before anything is downloaded: `need` (when known) plus
/// the store's reserve must be free. An unknown need is refused rather
/// than guessed, unless the caller gave a size.
pub fn check_space(need_bytes: Option<u64>, free_bytes: u64, reserve_gib: u64) -> Result<(), String> {
    let gib = |b: u64| b as f64 / GIB as f64;
    let reserve = reserve_gib * GIB;
    let Some(need) = need_bytes else {
        return Err("the download size is unknown; pass --size-gib with an upper bound".into());
    };
    if need.saturating_add(reserve) > free_bytes {
        return Err(format!(
            "not enough disk: needs {:.1} GiB and must leave {reserve_gib} GiB free, but only {:.1} GiB is free",
            gib(need),
            gib(free_bytes)
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_parse_and_refuse_anything_option_or_path_like() {
        assert_eq!(
            Source::parse("hf:Qwen/Qwen3.8-27B-FP8").unwrap(),
            Source::Hf { repo: "Qwen/Qwen3.8-27B-FP8".into(), revision: None }
        );
        assert_eq!(
            Source::parse("hf:org/name@6bc2f6e").unwrap(),
            Source::Hf { repo: "org/name".into(), revision: Some("6bc2f6e".into()) }
        );
        assert_eq!(Source::parse("ollama:qwen3-embedding:0.6b").unwrap(), Source::Ollama { tag: "qwen3-embedding:0.6b".into() });
        assert_eq!(Source::parse("hf:org/name@main").unwrap().to_string(), "hf:org/name@main");
        for bad in [
            "org/name", "hf:", "hf:name", "hf:a/b/c", "hf:--help/x", "hf:org/../etc", "hf:org/na me",
            "hf:org/name@--force", "ollama:", "ollama:-rm", "ollama:a b", "s3:bucket/x", "hf:org/name;rm",
        ] {
            assert!(Source::parse(bad).is_err(), "{bad}");
        }
        let hf = Source::parse("hf:a/b").unwrap();
        let ol = Source::parse("ollama:x").unwrap();
        assert!(hf.fits_store(StoreKind::HfCache) && hf.fits_store(StoreKind::Directory) && !hf.fits_store(StoreKind::Ollama));
        assert!(ol.fits_store(StoreKind::Ollama) && !ol.fits_store(StoreKind::HfCache));
    }

    #[test]
    fn the_hub_listing_is_summed_and_garbage_is_unknown() {
        let json = r#"[{"file": "a", "size": "791.0"}, {"file": "b", "size": "4.9K"},
                       {"file": "c", "size": "3.6M"}, {"file": "d", "size": "25.1G"}]"#;
        assert_eq!(hf_dry_run_bytes(json), Some(791 + 4_900 + 3_600_000 + 25_100_000_000));
        // Files already in the store are listed with "-": only the rest counts.
        assert_eq!(
            hf_dry_run_bytes(r#"[{"file": "a", "size": "-"}, {"file": "b", "size": "2.0M"}]"#),
            Some(2_000_000)
        );
        assert_eq!(hf_dry_run_bytes(r#"[{"file": "a", "size": "-"}]"#), Some(0));
        assert_eq!(hf_dry_run_bytes("[]"), None);
        assert_eq!(hf_dry_run_bytes(r#"[{"file": "a"}]"#), None);
        assert_eq!(hf_dry_run_bytes(r#"[{"file": "a", "size": "big"}]"#), None);
        assert_eq!(hf_dry_run_bytes("Traceback (most recent call last)"), None);
    }

    #[test]
    fn the_disk_check_keeps_the_reserve_and_refuses_an_unknown_size() {
        assert!(check_space(Some(50 * GIB), 80 * GIB, 20).is_ok());
        let e = check_space(Some(50 * GIB), 69 * GIB, 20).unwrap_err();
        assert!(e.contains("needs 50.0 GiB") && e.contains("only 69.0 GiB is free"), "{e}");
        assert!(check_space(None, 500 * GIB, 20).unwrap_err().contains("--size-gib"));
        assert!(check_space(Some(u64::MAX), 500 * GIB, 20).is_err(), "no overflow into a pass");
    }
}
