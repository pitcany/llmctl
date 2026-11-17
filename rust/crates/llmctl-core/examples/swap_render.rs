//! Render a llama-swap config through the lane records (round-trip proof
//! included): `cargo run -p llmctl-core --example swap_render -- config.yaml`.
fn main() {
    let path = std::env::args().nth(1).expect("usage: swap_render <config.yaml>");
    let text = std::fs::read_to_string(&path).expect("read config");
    match llmctl_core::swap::roundtrip(&text) {
        Ok((_, rendered)) => print!("{rendered}"),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
