//! Drawing. Reads [`App`] only; never asks an agent anything.

use llmctl_client::wire::{OperationView, PhaseStatus, PlanResponse, WorkloadView};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Frame;

use crate::app::{op_active, short, App, CatalogRow, Modal, RunRow, Screen};

const DIM: Style = Style::new().fg(Color::DarkGray);
const BOLD: Style = Style::new().add_modifier(Modifier::BOLD);
const STALE: Style = Style::new().fg(Color::Yellow);
const BAD: Style = Style::new().fg(Color::Red);
const GOOD: Style = Style::new().fg(Color::Green);
const SEL: Style = Style::new().add_modifier(Modifier::REVERSED);

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(1), Constraint::Length(2)])
        .split(area);
    draw_tabs(f, app, rows[0]);
    match app.screen {
        Screen::Hosts => draw_hosts(f, app, rows[1]),
        Screen::Run => draw_run(f, app, rows[1]),
        Screen::Ops => draw_ops(f, app, rows[1]),
        Screen::Logs => draw_logs(f, app, rows[1]),
        Screen::Gpu => draw_gpu(f, app, rows[1]),
        Screen::Catalog => draw_catalog(f, app, rows[1]),
        Screen::Audit => draw_audit(f, app, rows[1]),
    }
    draw_footer(f, app, rows[2]);
    if let Some(m) = &app.modal {
        draw_modal(f, app, m, area);
    }
    scrub(f);
}

/// Last pass over the frame: no cell may carry a control character to
/// the terminal. Agent text (engine log lines above all, but also errors,
/// phase details, audit states) can contain escape sequences; ratatui's
/// paragraph path writes them through as-is.
fn scrub(f: &mut Frame) {
    for cell in f.buffer_mut().content.iter_mut() {
        if cell.symbol().chars().any(char::is_control) {
            cell.set_symbol("?");
        }
    }
}

/// Agent text made safe to keep: tabs become spaces, other control
/// characters `?`.
pub fn clean(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\t' => ' ',
            c if c.is_control() => '?',
            c => c,
        })
        .collect()
}

fn draw_tabs(f: &mut Frame, app: &App, area: Rect) {
    let mut spans = vec![Span::styled(" llmctl-next ", BOLD)];
    for s in Screen::ALL {
        let label = format!(" {} {} ", s.key(), s.title());
        spans.push(if s == app.screen {
            Span::styled(label, SEL)
        } else {
            Span::raw(label)
        });
    }
    if app.read_only {
        spans.push(Span::styled(" READ-ONLY ", STALE));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let hints = match app.screen {
        Screen::Hosts => "↑↓ select · enter logs · r refresh · ? help · q quit",
        Screen::Run => "↑↓ select · enter run (presets, preview) · S stop · T restart (workloads) · q quit",
        Screen::Ops => "↑↓ select · C cancel · A acknowledge · r refresh · q quit",
        Screen::Logs => "↑↓ pick · enter open · f follow · PgUp/PgDn scroll · End newest · q quit",
        Screen::Gpu => "r refresh · ? help · q quit",
        Screen::Audit => "r refresh · ? help · q quit",
        Screen::Catalog => "↑↓ select · E enable · D disable · s scan · R record a scanned weight · q quit",
    };
    let msg = app.message.clone().unwrap_or_default();
    let lines = vec![Line::from(Span::styled(msg, BOLD)), Line::from(Span::styled(hints, DIM))];
    f.render_widget(Paragraph::new(lines), area);
}

fn boxed(title: String) -> Block<'static> {
    Block::default().borders(Borders::ALL).title(title)
}

// ---- Hosts --------------------------------------------------------------

/// One workload line, shared by Hosts and Run.
fn workload_line(app: &App, w: &WorkloadView, stale: bool) -> Line<'static> {
    let serves = match (&w.served_model, w.served_model_verified) {
        (Some(m), true) => format!("{m} ✓"),
        (Some(m), false) => format!("{m} ~"),
        (None, _) => "-".into(),
    };
    let seen = match w.observed_at {
        Some(t) => app.ago(t),
        None => "never".into(),
    };
    let fresh_mark = if stale {
        " [stale]"
    } else if w.observation_unknown {
        " [unknown]"
    } else {
        ""
    };
    let ready_style = match (stale, w.readiness.to_ascii_lowercase().replace('_', "").as_str()) {
        (true, _) => STALE,
        (_, "ready") => GOOD,
        (_, "unknown") => DIM,
        (_, "notready" | "failed") if w.process.eq_ignore_ascii_case("running") => BAD,
        _ => Style::new(),
    };
    Line::from(vec![
        Span::raw(format!("  {:<20} ", truncate(&w.name, 20))),
        Span::styled(format!("{:<9}", w.process), if stale { STALE } else { Style::new() }),
        Span::styled(format!("{:<10}", w.readiness), ready_style),
        Span::raw(format!("intent {:<16} ", truncate(&fmt_intent(&w.intent), 16))),
        Span::raw(format!("{:<30} ", truncate(&serves, 30))),
        Span::styled(format!("{:<8} ", w.ownership), DIM),
        Span::styled(format!("{seen}{fresh_mark}"), if stale { STALE } else { DIM }),
    ])
}

fn host_header(app: &App, h: &crate::fleet::HostSnapshot) -> Line<'static> {
    let agent = h.agent_version.clone().unwrap_or_else(|| "?".into());
    match (&h.fresh, &h.cached) {
        (Some(s), _) => Line::from(vec![
            Span::styled(format!("{:<12}", h.name), BOLD),
            Span::raw(format!(
                " agent {agent}  {}  seen {}  revision {}  {} workload(s)",
                s.platform,
                app.ago(s.generated_at),
                s.inventory_revision,
                s.workloads.len()
            )),
        ]),
        (None, cached) => {
            let since = match h.last_seen {
                Some(t) => format!("last seen {}", app.ago(t)),
                None => "never seen".into(),
            };
            let what = if cached.is_some() {
                "last observations shown"
            } else {
                "no cached observations"
            };
            Line::from(vec![
                Span::styled(format!("{:<12}", h.name), BOLD),
                Span::styled(" STALE ", STALE.add_modifier(Modifier::REVERSED)),
                Span::styled(format!(" {since} · {what}"), STALE),
            ])
        }
    }
}

/// Items for a host list, and which item each selectable row is.
fn host_items(app: &App, with_children: bool) -> (Vec<ListItem<'static>>, Vec<usize>) {
    let mut items = Vec::new();
    let mut selectable = Vec::new();
    if app.hosts.is_empty() {
        items.push(ListItem::new(Line::styled("asking the agents…", DIM)));
    }
    for h in &app.hosts {
        items.push(ListItem::new(host_header(app, h)));
        if let Some(err) = &h.error {
            items.push(ListItem::new(Line::styled(format!("  ({err})"), STALE)));
        }
        if let Some(s) = h.shown() {
            for w in &s.workloads {
                selectable.push(items.len());
                items.push(ListItem::new(workload_line(app, w, h.is_stale())));
                if with_children {
                    for m in app.loaded_under(&h.name, &w.name) {
                        items.push(ListItem::new(Line::styled(format!("      └ loaded {m}"), DIM)));
                    }
                }
            }
        }
    }
    (items, selectable)
}

fn draw_hosts(f: &mut Frame, app: &App, area: Rect) {
    let (items, selectable) = host_items(app, true);
    let mut state = ListState::default().with_selected(selectable.get(app.hosts_sel).copied());
    let list = List::new(items).block(boxed(" Hosts ".into())).highlight_style(SEL);
    f.render_stateful_widget(list, area, &mut state);
}

// ---- Run ----------------------------------------------------------------

/// A preset line's state word, the CLI `preset list`'s vocabulary.
fn preset_state(p: &llmctl_client::catalog::PresetSummary) -> (&'static str, Style) {
    match (&p.link, &p.link_error) {
        (Some(l), _) if l.active => ("serving", GOOD),
        (Some(l), _) if l.profile_env.is_none() => ("no-profile", STALE),
        (Some(l), _) if l.drift.is_empty() => ("ready", Style::new()),
        (Some(_), _) => ("drift", BAD),
        (None, Some(_)) => ("unlinked", STALE),
        (None, None) => ("-", DIM),
    }
}

fn draw_run(f: &mut Frame, app: &App, area: Rect) {
    let rows = app.run_rows();
    let mut items: Vec<ListItem<'static>> = Vec::new();
    let mut selectable: Vec<usize> = Vec::new();
    // Presets first (per host that answered), then workloads — a header
    // for each section on each host.
    let mut last: Option<(&str, bool)> = None; // (host, is_preset)
    for row in &rows {
        match row {
            RunRow::Preset { host, preset, stale } => {
                if last != Some((host, true)) {
                    let h = app.hosts.iter().find(|h| h.name == *host);
                    let note = match (&app.presets, h.map(|h| h.is_stale())) {
                        (_, Some(true)) => " STALE", // unreachable in practice: filtered
                        (p, _) if p.is_empty() => " asking…",
                        _ => "",
                    };
                    items.push(ListItem::new(Line::styled(format!(" presets on {host}{note}"), BOLD)));
                    last = Some((host, true));
                }
                let (state, style) = preset_state(preset);
                let line_style = if *stale { STALE } else { Style::new() };
                selectable.push(items.len());
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(format!("  {:<22} ", truncate(&preset.alias, 22)), line_style),
                    Span::styled(format!("{:<22} ", truncate(&preset.served_name, 22)), line_style),
                    Span::styled(format!("tp {:<2} ctx {:<6}", preset.tensor_parallel_size, preset.max_model_len), DIM),
                    Span::styled(format!(" {state:<10}"), style),
                ])));
            }
            RunRow::Workload(w) => {
                if last != Some((w.host, false)) {
                    items.push(ListItem::new(Line::styled(format!(" workloads on {}", w.host), BOLD)));
                    last = Some((w.host, false));
                }
                selectable.push(items.len());
                items.push(ListItem::new(workload_line(app, w.workload, w.stale)));
            }
        }
    }
    if items.is_empty() {
        let preset_errs: Vec<String> = app
            .presets
            .iter()
            .filter_map(|h| h.result.as_ref().err().map(|e| format!("{}: {e}", h.host)))
            .collect();
        if preset_errs.is_empty() {
            items.push(ListItem::new(Line::styled("asking the agents…", DIM)));
        } else {
            for e in preset_errs {
                items.push(ListItem::new(Line::styled(e, STALE)));
            }
        }
    }
    let mut state = ListState::default().with_selected(selectable.get(app.run_sel).copied());
    let list = List::new(items).block(boxed(" Run: presets and workloads ".into())).highlight_style(SEL);
    f.render_stateful_widget(list, area, &mut state);
}

// ---- Operations ---------------------------------------------------------

fn phase_mark(s: PhaseStatus) -> &'static str {
    match s {
        PhaseStatus::Pending => " ",
        PhaseStatus::Running => "▶",
        PhaseStatus::EffectPending => "…",
        PhaseStatus::Done => "✓",
        PhaseStatus::Failed => "✗",
        PhaseStatus::Skipped => "-",
    }
}

fn op_line(app: &App, host: &str, op: &OperationView) -> Line<'static> {
    let done = op
        .phases
        .iter()
        .filter(|p| matches!(p.status, PhaseStatus::Done | PhaseStatus::Skipped))
        .count();
    let status_style = match op.status.as_str() {
        "succeeded" => GOOD,
        "failed" | "needs_attention" | "interrupted" => BAD,
        "executing" | "admitted" | "cancel_requested" => STALE,
        _ => Style::new(),
    };
    let ack = if !op.acknowledged && matches!(op.status.as_str(), "needs_attention" | "interrupted") {
        "  [ack pending]"
    } else {
        ""
    };
    Line::from(vec![
        Span::raw(format!("{:<12} ", truncate(host, 12))),
        Span::styled(app.fmt_time(op.created_at, "%m-%d %H:%M:%S "), DIM),
        Span::raw(format!("{:<8} ", short(&op.id))),
        Span::raw(format!("{:<10} {:<20} ", op.kind.to_string(), truncate(&op.workload, 20))),
        Span::styled(format!("{:<16}", op.status.as_str()), status_style),
        Span::raw(format!(" phases {done}/{}", op.phases.len())),
        Span::styled(ack.to_string(), BAD),
    ])
}

fn op_detail(app: &App, host: &str, op: &OperationView) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::styled(format!("operation {} on {host}", op.id), BOLD),
        Line::raw(format!("key {}", op.idempotency_key)),
        Line::raw(format!(
            "{} {} — {}  (updated {})",
            op.kind,
            op.workload,
            op.status,
            app.ago(op.updated_at)
        )),
    ];
    for p in &op.phases {
        lines.push(Line::raw(format!(
            "  {} {:>2}. {}{}",
            phase_mark(p.status),
            p.index + 1,
            p.label,
            p.detail.as_deref().map(|d| format!(" — {d}")).unwrap_or_default()
        )));
    }
    if let Some(r) = &op.result {
        lines.push(Line::raw(format!("result: {}", compact(r))));
    }
    if let Some(r) = &op.recovery {
        lines.push(Line::styled(format!("recovery: {}", compact(r)), STALE));
    }
    if !op.acknowledged && matches!(op.status.as_str(), "needs_attention" | "interrupted") {
        lines.push(Line::styled("not acknowledged: A acknowledges (after a look)", BAD));
    }
    lines
}

fn draw_ops(f: &mut Frame, app: &App, area: Rect) {
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);
    let mut items: Vec<ListItem> = app.ops.iter().map(|(h, o)| ListItem::new(op_line(app, h, o))).collect();
    let errors = app.ops_errors.len();
    for (h, e) in &app.ops_errors {
        items.push(ListItem::new(Line::styled(format!("{h}: {e}"), STALE)));
    }
    if items.is_empty() {
        items.push(ListItem::new(Line::styled("no operations", DIM)));
    }
    let skipped = stale_note(app);
    let title = format!(" Operations ({}){skipped} ", app.ops.len());
    let mut state = ListState::default().with_selected((!app.ops.is_empty()).then_some(app.ops_sel));
    let list = List::new(items).block(boxed(title)).highlight_style(SEL);
    f.render_stateful_widget(list, parts[0], &mut state);
    let detail = match app.ops.get(app.ops_sel) {
        Some((h, o)) => op_detail(app, h, o),
        None if errors > 0 => vec![Line::styled("some hosts did not answer (see above)", STALE)],
        None => vec![Line::styled("nothing selected", DIM)],
    };
    f.render_widget(
        Paragraph::new(detail).block(boxed(" Details ".into())).wrap(Wrap { trim: false }),
        parts[1],
    );
}

/// " · m5 STALE, skipped" for screens that ask only reachable hosts.
fn stale_note(app: &App) -> String {
    let stale: Vec<&str> = app.hosts.iter().filter(|h| h.is_stale()).map(|h| h.name.as_str()).collect();
    if stale.is_empty() {
        String::new()
    } else {
        format!(" · {} STALE, skipped", stale.join(", "))
    }
}

// ---- Logs ---------------------------------------------------------------

fn draw_logs(f: &mut Frame, app: &App, area: Rect) {
    let parts = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(40), Constraint::Min(10)])
        .split(area);
    let targets = app.log_targets();
    let mut items: Vec<ListItem> = targets
        .iter()
        .map(|t| {
            let label = match &t.lane {
                Some(l) => format!("    lane {}", truncate(l, 24)),
                None => format!("{:<12} {}", truncate(&t.host, 12), truncate(&t.workload, 24)),
            };
            let style = if app.logs.target.as_ref() == Some(t) { BOLD } else { Style::new() };
            ListItem::new(Line::styled(label, style))
        })
        .collect();
    if items.is_empty() {
        items.push(ListItem::new(Line::styled("no workloads yet", DIM)));
    }
    let mut state = ListState::default().with_selected((!targets.is_empty()).then_some(app.logs.picker_sel));
    f.render_stateful_widget(
        List::new(items).block(boxed(" Pick ".into())).highlight_style(SEL),
        parts[0],
        &mut state,
    );

    let title = match &app.logs.target {
        Some(t) => format!(
            " {} · {} ",
            t.label(),
            if app.logs.follow { "following" } else { "paused" }
        ),
        None => " Logs ".to_string(),
    };
    let inner_h = parts[1].height.saturating_sub(2) as usize;
    let mut body: Vec<Line> = Vec::new();
    if let Some(e) = &app.logs.error {
        body.push(Line::styled(format!("error: {e}"), BAD));
    }
    if app.logs.target.is_none() {
        body.push(Line::styled("enter opens the picked log; f toggles follow", DIM));
    } else if app.logs.lines.is_empty() && app.logs.error.is_none() {
        body.push(Line::styled(
            if app.logs.in_flight { "reading…" } else { "(no lines yet)" },
            DIM,
        ));
    }
    let room = inner_h.saturating_sub(body.len());
    let end = app.logs.lines.len().saturating_sub(app.logs.scroll);
    let start = end.saturating_sub(room);
    body.extend(app.logs.lines[start..end].iter().map(|l| Line::raw(l.clone())));
    f.render_widget(Paragraph::new(body).block(boxed(title)), parts[1]);
}

// ---- GPUs / memory ------------------------------------------------------

fn gib(mib: u64) -> String {
    format!("{:.1} GiB", mib as f64 / 1024.0)
}

fn draw_gpu(f: &mut Frame, app: &App, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();
    if app.gpu.is_empty() {
        lines.push(Line::styled("asking the agents…", DIM));
    }
    for h in &app.hosts {
        if h.is_stale() {
            lines.push(Line::from(vec![
                Span::styled(format!("{:<12}", h.name), BOLD),
                Span::styled(" STALE — not asked until it answers the status call again", STALE),
            ]));
            continue;
        }
        let Some(g) = app.gpu.iter().find(|g| g.host == h.name) else {
            if !app.gpu.is_empty() {
                lines.push(Line::styled(format!("{:<12} (not asked yet)", h.name), DIM));
            }
            continue;
        };
        lines.push(Line::styled(g.host.clone(), BOLD));
        if let Some(e) = &g.error {
            lines.push(Line::styled(format!("  {e}"), STALE));
            continue;
        }
        // Cards: occupancy from the status answer, free memory from vram.
        let snap = app.hosts.iter().find(|h| h.name == g.host).and_then(|h| h.shown());
        let free = g.vram.as_ref().and_then(|v| v.as_ref().ok()).and_then(|v| v.free_mib.clone());
        // No VRAM endpoint (the Macs' unified memory): no per-card free
        // column at all — the lane budget lines below are the memory
        // picture there (M5e leftover: no more `free ?` per card).
        let mut vram_note_shown = false;
        if let Some(s) = snap {
            for c in &s.gpus {
                if let Some(err) = &c.error {
                    lines.push(Line::styled(format!("  gpu {:<3} occupancy unknown: {err}", c.index), STALE));
                    continue;
                }
                let free_s = match &free {
                    None => {
                        if !vram_note_shown {
                            vram_note_shown = true;
                            lines.push(Line::styled(
                                "  (no per-card free VRAM on this host; the memory budget below is the picture)",
                                DIM,
                            ));
                        }
                        "".to_string()
                    }
                    Some(m) => match m.get(&c.index) {
                        Some(mib) => format!("free {:<9}", gib(*mib)),
                        // The map answered but not for this card: a real unknown.
                        None => "free ?        ".into(),
                    },
                };
                let enrolled = if c.enrolled.is_empty() { "-".into() } else { c.enrolled.join(",") };
                let foreign = if c.foreign.is_empty() {
                    "-".to_string()
                } else {
                    c.foreign
                        .iter()
                        .map(|o| match &o.unit {
                            Some(u) => format!("pid {} ({u}, {})", o.pid, gib(o.memory_mib)),
                            None => format!("pid {} (no unit, {})", o.pid, gib(o.memory_mib)),
                        })
                        .collect::<Vec<_>>()
                        .join("; ")
                };
                let fstyle = if c.foreign.is_empty() { Style::new() } else { BAD };
                lines.push(Line::from(vec![
                    Span::raw(format!("  gpu {:<3} {free_s} enrolled {:<24} ", c.index, truncate(&enrolled, 24))),
                    Span::styled(format!("foreign {foreign}"), fstyle),
                ]));
            }
        }
        if let Some(Ok(v)) = &g.vram {
            for r in &v.records {
                let per = r
                    .per_card_mib
                    .iter()
                    .map(|(c, m)| format!("{} on gpu {c}", gib(*m)))
                    .collect::<Vec<_>>()
                    .join(", ");
                lines.push(Line::styled(
                    format!("  measured {:<16} {:<30} {per}", truncate(&r.workload, 16), truncate(&r.model, 30)),
                    DIM,
                ));
            }
        }
        for l in &g.lanes {
            let running = match &l.running {
                Some(r) if r.is_empty() => "none".to_string(),
                Some(r) => r.join(", "),
                None => "? (llama-swap did not answer)".into(),
            };
            let sum: f64 = l
                .running
                .iter()
                .flatten()
                .filter_map(|lane| l.peaks.get(lane))
                .sum();
            let budget = match l.budget_gib {
                Some(b) => format!("budget {b} GiB, loaded peaks {sum:.1} GiB"),
                None => "no memory budget set".into(),
            };
            let style = if l.budget_gib.is_some_and(|b| sum > b as f64) { BAD } else { Style::new() };
            lines.push(Line::styled(format!("  {} (llama-swap): {budget}", l.workload), style));
            lines.push(Line::raw(format!("    loaded lanes: {running}")));
            for (set, gib) in &l.over {
                lines.push(Line::styled(format!("    over budget: {set} ({gib} GiB)"), BAD));
            }
        }
        for e in &g.engines {
            let loaded = e.loaded.clone().unwrap_or_default();
            lines.push(Line::raw(format!(
                "  {} ({}): {} loaded{}",
                e.workload,
                e.engine_kind,
                loaded.len(),
                if loaded.is_empty() { String::new() } else { format!(" — {}", loaded.join(", ")) }
            )));
        }
    }
    f.render_widget(
        Paragraph::new(lines).block(boxed(" GPUs / memory ".into())).wrap(Wrap { trim: false }),
        area,
    );
}

// ---- Catalog ------------------------------------------------------------

fn draw_catalog(f: &mut Frame, app: &App, area: Rect) {
    let parts = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);
    let rows = app.catalog_rows();
    let mut items: Vec<ListItem<'static>> = Vec::new();
    let mut selectable: Vec<usize> = Vec::new();
    if app.catalog.is_empty() && app.scans.is_empty() {
        items.push(ListItem::new(Line::styled("asking the agents…", DIM)));
    }
    for h in &app.catalog {
        let line_style = if app.host_stale(&h.host) { STALE } else { Style::new() };
        match &h.result {
            Err(e) => items.push(ListItem::new(Line::styled(format!("{}: {e}", h.host), STALE))),
            Ok(c) => {
                let mut tail = format!("rev {} · {} models", c.revision, c.models.len());
                if c.disabled_hidden > 0 {
                    tail.push_str(&format!(" · {} disabled", c.disabled_hidden));
                }
                if c.unmapped > 0 {
                    tail.push_str(&format!(" · {} bundle-only", c.unmapped));
                }
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(format!("{:<12}", h.host), BOLD),
                    Span::styled(format!(" {tail}"), line_style),
                ])));
                let mut last_store: Option<String> = None;
                for m in &c.models {
                    let store = m.store.clone().unwrap_or_else(|| "(no store)".into());
                    if last_store.as_deref() != Some(store.as_str()) {
                        items.push(ListItem::new(Line::styled(format!("  [{store}]"), DIM)));
                        last_store = Some(store);
                    }
                    let style = if m.enabled { line_style } else { DIM };
                    let mark = if m.enabled { "" } else { "  (disabled)" };
                    selectable.push(items.len());
                    items.push(ListItem::new(Line::styled(
                        format!(
                            "    {:<34} {:<10} {}{}",
                            truncate(&m.name, 34),
                            m.engine,
                            m.source.as_deref().or(m.path.as_deref()).unwrap_or("-"),
                            mark
                        ),
                        style,
                    )));
                }
            }
        }
    }
    for (host, scan) in &app.scans {
        let (title, lines): (String, Vec<Line<'static>>) = match scan {
            Ok(s) => {
                let fresh: Vec<_> = s.found.iter().filter(|f| f.record.is_none()).collect();
                let t = format!("scan on {host}: {} weights, {} not cataloged", s.found.len(), fresh.len());
                let mut out = Vec::new();
                for fdat in fresh {
                    selectable.push(items.len());
                    out.push(Line::styled(
                        format!("  + {:<34} {:<9} {} · R records", truncate(&fdat.source, 34), fdat.engine, fdat.store),
                        GOOD,
                    ));
                }
                for m in &s.missing {
                    out.push(Line::styled(
                        format!("  ! gone: {} ({})", m.name, m.path.as_deref().unwrap_or("-")),
                        BAD,
                    ));
                }
                (t, out)
            }
            Err(e) => (format!("scan on {host} failed"), vec![Line::styled(format!("  {e}"), STALE)]),
        };
        items.push(ListItem::new(Line::styled(format!(" {title}"), BOLD)));
        items.extend(lines.into_iter().map(ListItem::new));
    }
    let title = format!(" Catalog ({} rows){} ", rows.len(), stale_note(app));
    let mut state = ListState::default().with_selected(selectable.get(app.catalog_sel).copied());
    let list = List::new(items).block(boxed(title)).highlight_style(SEL);
    f.render_stateful_widget(list, parts[0], &mut state);

    let detail = match rows.get(app.catalog_sel) {
        Some(CatalogRow::Model { host, model, .. }) => {
            let mut out = vec![
                Line::styled(format!("{} · {}", host, model.name), BOLD),
                Line::raw(format!("id {} · engine {} · origin {}", model.id, model.engine, model.origin)),
                Line::raw(format!(
                    "store {} · source {}",
                    model.store.as_deref().unwrap_or("-"),
                    model.source.as_deref().unwrap_or("-")
                )),
            ];
            if let Some(p) = &model.path {
                out.push(Line::raw(format!("path {p}")));
            }
            out.push(Line::raw(if model.enabled {
                "enabled (D disables)".to_string()
            } else {
                "disabled · hidden from `model list`, exported inactive (E enables)".to_string()
            }));
            for (k, v) in &model.extra {
                out.push(Line::styled(format!("{k}: {v}"), DIM));
            }
            out
        }
        Some(CatalogRow::Found { host, found, .. }) => vec![
            Line::styled(format!("{host} · scanned weight"), BOLD),
            Line::raw(format!("{} · engine {}", found.source, found.engine)),
            Line::raw(format!("store {} · {}", found.store, found.path)),
            Line::styled("R records it in the catalog", GOOD),
        ],
        None if app.catalog.is_empty() => vec![Line::styled("asking the agents…", DIM)],
        None => vec![Line::styled("nothing selected", DIM)],
    };
    f.render_widget(
        Paragraph::new(detail).block(boxed(" Details ".into())).wrap(Wrap { trim: false }),
        parts[1],
    );
}

// ---- Audit --------------------------------------------------------------

/// One comparison state for one line: `running pid=100 model=qwen`.
pub fn fmt_seen(raw: &str) -> String {
    let v: serde_json::Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return raw.to_string(),
    };
    let mut parts = Vec::new();
    if let Some(p) = v.get("process").and_then(|p| p.as_str()) {
        parts.push(p.to_string());
    }
    for key in ["pid", "model"] {
        if let Some(x) = v.get(key) {
            if !x.is_null() {
                let s = match x {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                parts.push(format!("{key}={s}"));
            }
        }
    }
    if let Some(lanes) = v.get("lanes").and_then(|l| l.as_array()) {
        let names: Vec<String> = lanes.iter().filter_map(|l| l.as_str().map(String::from)).collect();
        parts.push(format!("lanes=[{}]", names.join(",")));
    }
    if parts.is_empty() {
        raw.to_string()
    } else {
        parts.join(" ")
    }
}

fn draw_audit(f: &mut Frame, app: &App, area: Rect) {
    let mut items: Vec<ListItem> = Vec::new();
    let mut selectable = Vec::new();
    let mut total = 0usize;
    if app.audit.is_empty() {
        items.push(ListItem::new(Line::styled("asking the agents…", DIM)));
    }
    for h in &app.audit {
        match &h.result {
            Err(e) => items.push(ListItem::new(Line::styled(format!("{}: {e}", h.host), STALE))),
            Ok(fs) if fs.is_empty() => {
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(format!("{:<12}", h.host), BOLD),
                    Span::styled(" no foreign writes", GOOD),
                ])))
            }
            Ok(fs) => {
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(format!("{:<12}", h.host), BOLD),
                    Span::styled(format!(" {} foreign write(s)", fs.len()), BAD),
                ])));
                total += fs.len();
                for x in fs {
                    selectable.push(items.len());
                    items.push(ListItem::new(Line::raw(format!(
                        "  {}  {:<16} {:<10} {} -> {}{}",
                        app.fmt_time(x.observed_at, "%Y-%m-%d %H:%M:%S"),
                        truncate(&x.workload, 16),
                        x.what,
                        fmt_seen(&x.before_state),
                        fmt_seen(&x.after_state),
                        x.actor.as_deref().map(|a| format!("  actor={a}")).unwrap_or_default()
                    ))));
                }
            }
        }
    }
    let title = format!(" Audit: foreign writes ({total}){} ", stale_note(app));
    let mut state = ListState::default().with_selected(selectable.get(app.audit_sel).copied());
    f.render_stateful_widget(List::new(items).block(boxed(title)).highlight_style(SEL), area, &mut state);
}

// ---- modals -------------------------------------------------------------

fn centered(area: Rect, w: u16, h: u16) -> Rect {
    let w = w.min(area.width.saturating_sub(2));
    let h = h.min(area.height.saturating_sub(2));
    Rect { x: area.x + (area.width - w) / 2, y: area.y + (area.height - h) / 2, width: w, height: h }
}

pub fn plan_lines(p: &PlanResponse, host: &str) -> Vec<Line<'static>> {
    let plan = &p.plan;
    let mut lines = vec![
        Line::styled(format!("plan: {} {} on {host} @ inventory revision {}", plan.kind, plan.target, plan.revision), BOLD),
        Line::raw(format!(
            "expected model: {}  gpus: {}",
            plan.expected_model,
            plan.affected_gpus.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(",")
        )),
    ];
    if p.stop_targets.is_empty() {
        lines.push(Line::styled("stops nothing", GOOD));
    } else {
        lines.push(Line::styled(format!("STOP TARGETS: {}", p.stop_targets.join(", ")), BAD));
    }
    lines.push(Line::raw("steps:"));
    for (i, s) in plan.steps.iter().enumerate() {
        lines.push(Line::raw(format!(
            "  {:>2}. {}{}",
            i + 1,
            s.label(),
            if s.is_external() { "" } else { "  (internal)" }
        )));
    }
    if !plan.rollback.is_empty() {
        lines.push(Line::raw("rollback (on failure):"));
        for (i, s) in plan.rollback.iter().enumerate() {
            lines.push(Line::raw(format!("  {:>2}. {}", i + 1, s.label())));
        }
    }
    for n in &p.notes {
        lines.push(Line::styled(format!("note: {n}"), DIM));
    }
    lines
}

fn draw_modal(f: &mut Frame, app: &App, m: &Modal, area: Rect) {
    let (title, lines): (String, Vec<Line>) = match m {
        Modal::Help => (
            " Help ".into(),
            vec![
                Line::styled("Screens", BOLD),
                Line::raw("  d Hosts · w Run · o Operations · l Logs · g GPUs/memory · m Catalog · a Audit"),
                Line::styled("Everywhere", BOLD),
                Line::raw("  r refresh now (unreachable hosts too) · ? help · q / ctrl-c quit · esc clear message"),
                Line::styled("Writes (each previewed or confirmed first)", BOLD),
                Line::raw("  Run: enter run (preset or workload) → plan preview → y admits · S stop · T restart (workloads)"),
                Line::raw("  Run on a drifted preset: s syncs the profile env, then the preview returns"),
                Line::raw("  Catalog: E enable · D disable · R record a scanned weight → y/n · s scans (a read)"),
                Line::raw("  Operations: C cancel · A acknowledge → y/n"),
                Line::raw(""),
                Line::raw("A host that does not answer within LLMCTL_NEXT_STATUS_TIMEOUT_S (default 10 s)"),
                Line::raw("shows its last observations as STALE and is retried every 60 s (r: now)."),
                Line::raw(""),
                Line::styled("any key closes", DIM),
            ],
        ),
        Modal::Confirm { title, lines, .. } => {
            let mut l: Vec<Line> = lines.iter().map(|s| Line::raw(s.clone())).collect();
            l.push(Line::raw(""));
            l.push(Line::styled("y confirm · n / esc cancel", BOLD));
            (format!(" {title} "), l)
        }
        Modal::Preview { host, kind, workload, alias, profile, plan, refused } => {
            let mut l = Vec::new();
            if let Some(r) = refused {
                l.push(Line::styled("REFUSED: the plan changed since the preview — nothing ran.", BAD));
                l.push(Line::styled(format!("  agent: {r}"), BAD));
                l.push(Line::styled("Review the fresh plan below and confirm again.", BAD));
                l.push(Line::raw(""));
            }
            let what = match alias {
                Some(a) => format!("preset {a}"),
                None => format!("{kind} {workload}"),
            };
            match plan {
                None => l.push(Line::styled(format!("planning {what} on {host}…"), DIM)),
                Some(Err(e)) => {
                    l.push(Line::styled(format!("no plan: {e}"), BAD));
                    if e.is(crate::api::DRIFT_CODE) && alias.is_some() {
                        l.push(Line::raw(""));
                        l.push(Line::styled("s syncs the profile env from the preset (a write), then plans again", BOLD));
                        l.push(Line::styled("n / esc close", BOLD));
                    } else {
                        l.push(Line::raw(""));
                        l.push(Line::styled("esc close", BOLD));
                    }
                }
                Some(Ok(p)) => {
                    if let (Some(a), Some(pr)) = (alias, profile) {
                        let name = pr.env_source.rsplit('/').next().unwrap_or(&pr.env_source);
                        l.push(Line::styled(
                            format!("preset {a}: switch {workload} to {name} (serves {})", pr.model),
                            DIM,
                        ));
                    } else if alias.is_some() {
                        l.push(Line::styled(
                            format!(
                                "preset {}: the active env already renders it; an ordinary run",
                                alias.as_deref().unwrap_or("")
                            ),
                            DIM,
                        ));
                    }
                    l.extend(plan_lines(p, host));
                    l.push(Line::raw(""));
                    l.push(Line::styled("y admit this plan · n / esc cancel", BOLD));
                }
            }
            (format!(" {what} on {host}: preview "), l)
        }
        Modal::Progress { host, kind, workload, op, error } => {
            let mut l = Vec::new();
            match (op, error) {
                (_, Some(e)) => l.push(Line::styled(format!("not admitted: {e}"), BAD)),
                (None, None) => l.push(Line::styled("admitting…", DIM)),
                (Some(o), None) => {
                    l.extend(op_detail(app, host, o));
                    l.push(Line::raw(""));
                    if op_active(o) {
                        l.push(Line::styled("following… esc closes (the operation keeps running; see o)", DIM));
                    }
                }
            }
            if op.as_ref().is_some_and(|o| !op_active(o)) || error.is_some() {
                l.push(Line::styled("esc / enter close", BOLD));
            }
            (format!(" {kind} {workload} on {host} "), l)
        }
    };
    let h = (lines.len() as u16 + 2).max(5);
    let rect = centered(area, 96, h);
    f.render_widget(Clear, rect);
    f.render_widget(Paragraph::new(lines).block(boxed(title)).wrap(Wrap { trim: false }), rect);
}

// ---- helpers ------------------------------------------------------------

/// The recorded intent for one cell: the agent sends JSON for a desired
/// state (`{"state":"stopped","manual":true}`) and a bare word otherwise.
pub fn fmt_intent(raw: &str) -> String {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(raw) else {
        return raw.to_string();
    };
    match v.get("state").and_then(|s| s.as_str()) {
        Some(state) if v.get("manual").and_then(|m| m.as_bool()) == Some(true) => format!("{state} (manual)"),
        Some(state) => state.to_string(),
        None => raw.to_string(),
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(n.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn compact(v: &serde_json::Value) -> String {
    let s = v.to_string();
    if s.len() > 400 {
        format!("{}…", &s[..s.char_indices().take_while(|(i, _)| *i < 400).last().map(|(i, _)| i).unwrap_or(0)])
    } else {
        s
    }
}
