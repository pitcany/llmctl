//! Running one companion hook (M4c item 5): an argv without a shell, in a
//! small fixed environment, killed when its time is up, its output kept to
//! a bounded tail for the journal.
//!
//! Hooks come from the inventory file only. No API or CLI verb registers
//! one, so the control socket cannot be used to run a command of the
//! caller's choosing.

use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncReadExt;

/// How much of a hook's combined output the journal keeps (the tail).
pub const MAX_OUTPUT_BYTES: usize = 2000;

/// Run `argv` for `workload`. `Ok(report)` on exit 0; `Err(report)` on a
/// non-zero exit, a spawn failure or a timeout. The report is one line of
/// outcome plus the output tail.
pub async fn run(
    workload: &str,
    name: &str,
    phase: &str,
    argv: &[String],
    timeout: Duration,
) -> Result<String, String> {
    let Some((program, args)) = argv.split_first() else {
        return Err(format!("hook {name} has no command"));
    };
    let mut cmd = tokio::process::Command::new(program);
    cmd.args(args)
        .env_clear()
        .envs(std::env::vars().filter(|(k, _)| {
            matches!(k.as_str(), "HOME" | "USER" | "LOGNAME" | "PATH" | "XDG_RUNTIME_DIR" | "DBUS_SESSION_BUS_ADDRESS" | "LANG")
        }))
        .env("LLMCTL_WORKLOAD", workload)
        .env("LLMCTL_HOOK", name)
        .env("LLMCTL_HOOK_PHASE", phase)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("hook {name}: cannot run {program}: {e}"))?;
    let mut stdout = child.stdout.take().expect("piped stdout");
    let mut stderr = child.stderr.take().expect("piped stderr");
    // Drain both pipes while waiting, so a chatty hook cannot block on a
    // full pipe; only the tail is kept.
    let collect = async {
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let (a, b, status) = tokio::join!(
            read_tail(&mut stdout, &mut out),
            read_tail(&mut stderr, &mut err),
            child.wait()
        );
        let _ = (a, b);
        out.extend_from_slice(&err);
        (status, out)
    };
    match tokio::time::timeout(timeout, collect).await {
        Ok((Ok(status), output)) => {
            let text = tail_text(&output);
            if status.success() {
                Ok(format!("hook {name} ok{text}"))
            } else {
                Err(format!("hook {name} failed ({status}){text}"))
            }
        }
        Ok((Err(e), _)) => Err(format!("hook {name}: waiting for {program}: {e}")),
        // Dropping the future drops the child: kill_on_drop ends it.
        Err(_) => Err(format!("hook {name} timed out after {}s and was killed", timeout.as_secs())),
    }
}

/// Read to the end, keeping only the last [`MAX_OUTPUT_BYTES`].
async fn read_tail<R: tokio::io::AsyncRead + Unpin>(reader: &mut R, keep: &mut Vec<u8>) {
    let mut buf = [0u8; 4096];
    while let Ok(n) = reader.read(&mut buf).await {
        if n == 0 {
            break;
        }
        keep.extend_from_slice(&buf[..n]);
        if keep.len() > MAX_OUTPUT_BYTES {
            let cut = keep.len() - MAX_OUTPUT_BYTES;
            keep.drain(..cut);
        }
    }
}

fn tail_text(output: &[u8]) -> String {
    let start = output.len().saturating_sub(MAX_OUTPUT_BYTES);
    let text = String::from_utf8_lossy(&output[start..]);
    let text = text.trim();
    if text.is_empty() {
        String::new()
    } else {
        format!(": {text}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(script: &str) -> Vec<String> {
        vec!["/bin/sh".into(), "-c".into(), script.into()]
    }

    #[tokio::test]
    async fn a_hook_reports_its_exit_and_a_bounded_tail_of_its_output() {
        let t = Duration::from_secs(10);
        let ok = run("coder", "h", "pre_start", &sh("echo out; echo err >&2"), t).await.unwrap();
        assert!(ok.starts_with("hook h ok: ") && ok.contains("out") && ok.contains("err"), "{ok}");

        let failed = run("coder", "h", "pre_start", &sh("echo nope; exit 3"), t).await.unwrap_err();
        assert!(failed.contains("failed") && failed.contains("nope"), "{failed}");

        // A megabyte of output: only the tail is kept, and nothing blocks.
        let big = run("coder", "h", "pre_start", &sh("head -c 1000000 /dev/zero | tr '\\0' x; echo END"), t)
            .await
            .unwrap();
        assert!(big.len() < MAX_OUTPUT_BYTES + 100 && big.ends_with("END"), "{} bytes", big.len());

        let missing = run("coder", "h", "pre_start", &["/nonexistent/hook".to_string()], t).await.unwrap_err();
        assert!(missing.contains("cannot run /nonexistent/hook"), "{missing}");
    }

    #[tokio::test]
    async fn a_hook_gets_its_context_and_not_the_daemons_whole_environment() {
        std::env::set_var("LLMCTL_TEST_SECRET_TOKEN", "do-not-leak");
        let out = run(
            "coder",
            "ctx",
            "post_stop",
            &sh("echo $LLMCTL_WORKLOAD/$LLMCTL_HOOK/$LLMCTL_HOOK_PHASE; env | grep -c SECRET_TOKEN || true"),
            Duration::from_secs(10),
        )
        .await
        .unwrap();
        assert!(out.contains("coder/ctx/post_stop"), "{out}");
        assert!(!out.contains("do-not-leak") && out.trim_end().ends_with('0'), "{out}");
    }

    #[tokio::test]
    async fn a_hook_that_overruns_is_killed() {
        let started = std::time::Instant::now();
        let err = run("coder", "slow", "pre_stop", &sh("sleep 30"), Duration::from_millis(300)).await.unwrap_err();
        assert!(err.contains("timed out"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
