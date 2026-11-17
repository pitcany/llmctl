//! The SSH stdio bridge transport (RUST-M3-PLAN, design decision 1 and E):
//! the client spawns `ssh <alias> <remote> llmctl-next bridge`, and the
//! remote `bridge` subcommand relays its stdin/stdout to the agent's Unix
//! socket. The same HTTP/1.1 API and DTOs flow through the pipe unchanged;
//! SSH stays transport only (no remote shell logic, no listening port, no
//! stale forwarded sockets).

use std::pin::Pin;
use std::process::Stdio;
use std::sync::Arc;
use std::task::{Context, Poll};

use hyper::Uri;
use tokio::io::{AsyncRead, AsyncWrite, Join};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tower::Service;

/// The remote command prefix that starts the bridge on an enrolled host
/// (M3-D.1). The data dir is pinned so the bridge always reaches the
/// *agent's* socket, never a dev-default one; `~` expands in the remote
/// login shell. Overridable for drills.
pub fn remote_bin() -> String {
    std::env::var("LLMCTL_NEXT_REMOTE_BIN").unwrap_or_else(|_| {
        "~/.local/bin/llmctl-next --data-dir ~/.local/state/llmctl-next".into()
    })
}

/// One spawned `ssh` carrying the bridge protocol on its stdio. Killing
/// the child on drop guarantees no orphaned ssh after a client crash.
pub struct BridgeProcess {
    child: Child,
    io: Join<ChildStdout, ChildStdin>,
}

impl BridgeProcess {
    /// `ssh <alias> <remote> llmctl-next bridge` with piped stdio; ssh's
    /// own diagnostics stay on our stderr, never in the protocol stream.
    pub fn spawn_ssh(alias: &str) -> std::io::Result<Self> {
        let remote = format!("{} bridge", remote_bin());
        Self::spawn_ssh_cmd(alias, &remote)
    }

    /// Same, with an explicit remote command line (drills).
    pub fn spawn_ssh_cmd(alias: &str, remote: &str) -> std::io::Result<Self> {
        let mut child = Command::new("ssh")
            .arg(alias)
            .arg(remote)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        Ok(Self {
            io: tokio::io::join(stdout, stdin),
            child,
        })
    }

    pub fn id(&self) -> u32 {
        self.child.id().unwrap_or(0)
    }

    /// Kill the ssh process now (lost-ACK drill: the transport disappears
    /// right after a request is written).
    pub fn kill(&mut self) {
        let _ = self.child.start_kill();
    }
}

impl AsyncRead for BridgeProcess {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        AsyncRead::poll_read(Pin::new(&mut self.io), cx, buf)
    }
}

impl AsyncWrite for BridgeProcess {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        AsyncWrite::poll_write(Pin::new(&mut self.io), cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        AsyncWrite::poll_flush(Pin::new(&mut self.io), cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        AsyncWrite::poll_shutdown(Pin::new(&mut self.io), cx)
    }
}

/// hyper's connection marker: nothing extra to report (plain stdio pipe).
impl hyper_util::client::legacy::connect::Connection for BridgeProcess {
    fn connected(&self) -> hyper_util::client::legacy::connect::Connected {
        hyper_util::client::legacy::connect::Connected::new()
    }
}

/// Connector that dials a fresh `ssh` bridge for every new connection; the
/// HTTP client pools the healthy one and re-dials when it dies (a host
/// that slept and woke gets a new pipe transparently).
#[derive(Clone)]
pub struct BridgeConnector {
    pub alias: Arc<String>,
}

impl Service<Uri> for BridgeConnector {
    type Response = hyper_util::rt::TokioIo<BridgeProcess>;
    type Error = std::io::Error;
    type Future = Pin<
        Box<
            dyn std::future::Future<
                    Output = Result<hyper_util::rt::TokioIo<BridgeProcess>, std::io::Error>,
                > + Send,
        >,
    >;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _uri: Uri) -> Self::Future {
        let alias = self.alias.clone();
        Box::pin(async move {
            let process = BridgeProcess::spawn_ssh(&alias)?;
            Ok(hyper_util::rt::TokioIo::new(process))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn bridge_relay_through_cat() {
        // `cat` as the "remote bridge": bytes written to the pipe come
        // back unchanged, proving the stdio plumbing both directions.
        // (Dropping the join closes stdin — tokio's ChildStdin
        // poll_shutdown alone does not give cat its EOF.)
        let mut child = Command::new("cat")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .expect("spawn cat");
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let mut io = tokio::io::join(stdout, stdin);
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        io.write_all(b"ping").await.unwrap();
        let mut buf = [0u8; 4];
        io.read_exact(&mut buf).await.unwrap();
        assert_eq!(&buf, b"ping");
        drop(io);
        let _ = child.wait().await;
    }
}
