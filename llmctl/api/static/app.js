(function () {
  "use strict";

  // Poll cadence; short enough to feel live, long enough to spare the host.
  var POLL_MS = 5000;
  var JOB_POLL_MS = 1000;
  var TOKEN_KEY = "llmctl_token";
  var COLUMNS = ["#", "Model", "Engine", "Name", "Cards", "State", "Last load"];

  var pollTimer = null;
  var pollInFlight = false;
  var jobId = null;
  var lastDoneId = null;
  var jobBusy = false;
  var refreshQueued = false;
  var jobTimer = null;
  var jobRunning = false;
  var servingCount = 0;
  var confirmAction = null;

  // Shorthand lookup so the rest of the file stays readable.
  function $(id) {
    return document.getElementById(id);
  }

  // Removes all children of a node.
  function clear(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  // Reads the saved token; sessionStorage may throw when storage is blocked.
  function getToken() {
    try { return window.sessionStorage.getItem(TOKEN_KEY) || ""; } catch (e) { return ""; }
  }

  // Saves or clears the token for this tab only, never persisting beyond it.
  function setToken(value) {
    try {
      if (value) { window.sessionStorage.setItem(TOKEN_KEY, value); }
      else { window.sessionStorage.removeItem(TOKEN_KEY); }
    } catch (e) { /* storage blocked: the token will not be remembered */ }
  }

  // Builds an Error that carries the HTTP status so callers can react to 401.
  function apiError(message, status) {
    var err = new Error(message);
    err.status = status;
    return err;
  }

  // Turns a FastAPI error body into one readable string.
  function detailOf(data, fallback) {
    var d = data ? data.detail : undefined;
    return typeof d === "string" ? d : d !== undefined ? JSON.stringify(d) : fallback;
  }

  // Single entry point to the server; relative /swap paths keep it same-origin.
  function api(method, path, body) {
    if (path.indexOf("/swap") !== 0) { return Promise.reject(apiError("refusing non-/swap path", 0)); }
    var headers = { Accept: "application/json" };
    var token = getToken();
    if (token) { headers["Authorization"] = "Bearer " + token; }
    var init = { method: method, headers: headers };
    if (body !== undefined && body !== null) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    return fetch(path, init).then(
      function (res) {
        return res.text().then(function (text) {
          var data = null;
          try { data = text ? JSON.parse(text) : null; } catch (e) { data = null; }
          if (!res.ok) {
            throw apiError(detailOf(data, "HTTP " + res.status), res.status);
          }
          return data;
        });
      },
      function () {
        throw apiError("llmctl API unreachable", 0);
      }
    );
  }

  // Who put the current message up: "list" (the 5 s poll) or "action".
  var errorSource = null;

  // Shows a message in the alert area.
  function showError(message, source) {
    var area = $("error-area");
    area.textContent = message;
    area.hidden = false;
    errorSource = source || "action";
  }

  // Hides the alert area. With a source, only that source's message goes, so a
  // successful poll does not wipe a 409 from Run before anyone reads it.
  function clearError(source) {
    if (source && errorSource !== source) { return; }
    var area = $("error-area");
    area.textContent = "";
    area.hidden = true;
    errorSource = null;
  }

  // One shared promise, so concurrent 401s all resume on a single submit.
  var tokenPrompt = null;

  // Asks for a token; resolves once the user submits one, so the caller can retry.
  function askForToken() {
    if (tokenPrompt) { return tokenPrompt; }
    var form = $("token-form");
    form.hidden = false;
    tokenPrompt = new Promise(function (resolve) {
      form.addEventListener("submit", function onSubmit(ev) {
        ev.preventDefault();
        var value = $("token-input").value.trim();
        if (!value) { return; }
        form.removeEventListener("submit", onSubmit);
        setToken(value);
        $("token-input").value = "";
        form.hidden = true;
        tokenPrompt = null;
        resolve();
      });
    });
    return tokenPrompt;
  }

  // Calls api(); on 401 shows the token form and retries after submit.
  function apiWithAuth(method, path, body) {
    return api(method, path, body).catch(function (err) {
      if (err.status !== 401) { throw err; }
      setToken("");
      showError("Authentication required: enter the API token.");
      return askForToken().then(function () {
        clearError();
        return apiWithAuth(method, path, body);
      });
    });
  }

  // Mirrors lanes.format_duration so the UI and CLI agree: 25s, 2m 05s.
  function formatDuration(seconds) {
    if (seconds === null || seconds === undefined) { return "—"; }
    var total = Math.round(seconds);
    var secs = total % 60;
    return total >= 60 ? Math.floor(total / 60) + "m " + (secs < 10 ? "0" : "") + secs + "s" : secs + "s";
  }

  // The API sends cards as a string or a list; show either as one string.
  function cardsText(cards) {
    if (Array.isArray(cards)) { return cards.join(", "); }
    return cards === null || cards === undefined ? "" : String(cards);
  }

  // Small element factory; textContent only, so server data can never become markup.
  function el(tag, text, className) {
    var node = document.createElement(tag);
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    if (className) { node.className = className; }
    return node;
  }

  // A button that calls onClick; type=button keeps it from submitting forms.
  function button(label, onClick, disabled) {
    var b = el("button", label);
    b.type = "button";
    b.disabled = !!disabled;
    b.addEventListener("click", onClick);
    return b;
  }

  // Copies text to the clipboard and briefly confirms on the button.
  function copyText(text, btn) {
    // Shows a status label, then restores "Copy" (fixed, so double-clicks can't stick).
    var done = function (label) {
      btn.textContent = label;
      window.setTimeout(function () { btn.textContent = "Copy"; }, 1500);
    };
    if (!navigator.clipboard) { done("No clipboard"); return; }
    navigator.clipboard.writeText(text).then(
      function () { done("Copied"); },
      function () { done("Copy failed"); }
    );
  }

  // A "label value [Copy]" line used for endpoint, gateway and alias details.
  function copyLine(label, value) {
    var line = el("div", null, "copy-line");
    line.appendChild(el("span", label + " ", "copy-label"));
    line.appendChild(el("code", value));
    var btn = button("Copy", function () { copyText(value, btn); });
    line.appendChild(btn);
    return line;
  }

  // Fills the table header once, since the static HTML header may differ.
  function renderHeader() {
    var head = $("rows-table").tHead;
    clear(head);
    var tr = el("tr");
    COLUMNS.forEach(function (name) { tr.appendChild(el("th", name)); });
    tr.appendChild(el("th", "Action"));
    head.appendChild(tr);
  }

  // Builds the action cell: Stop for a serving row, Run for the rest.
  function actionCell(row, serving) {
    var td = el("td", null, "actions");
    if (serving) {
      td.appendChild(button("Stop", function () { startStop({ name: row.name }); }, jobRunning));
    } else {
      td.appendChild(button("Run", function () { startRun(row.name); }, jobRunning));
    }
    return td;
  }

  // Redraws the table body from the /swap payload. Endpoints and aliases live
  // in the Serving box only: repeating them here made serving rows four lines tall.
  function renderRows(rows) {
    var body = $("rows-body");
    clear(body);
    rows.forEach(function (row) {
      var serving = row.state === "serving";
      var tr = el("tr", null, serving ? "serving" : "");
      tr.appendChild(el("td", row.number));
      tr.appendChild(el("td", row.model));
      tr.appendChild(el("td", row.engine));
      tr.appendChild(el("td", row.name, "ident"));
      tr.appendChild(el("td", cardsText(row.cards)));
      tr.appendChild(el("td", row.state));
      tr.appendChild(el("td", formatDuration(row.last_load_s)));
      tr.appendChild(actionCell(row, serving));
      body.appendChild(tr);
    });
  }

  // Updates the serving box: who is serving, and whether Stop all is useful.
  function renderServing(rows, gatewayUrl) {
    var box = $("serving-summary");
    clear(box);
    var serving = rows.filter(function (r) { return r.state === "serving"; });
    if (!serving.length) { box.appendChild(el("p", "Nothing is serving.")); }
    serving.forEach(function (row) {
      var item = el("div", null, "serving-item");
      item.appendChild(el("strong", row.name));
      item.appendChild(el("span", " (" + row.model + ", cards " + cardsText(row.cards) + ")"));
      if (row.endpoint) { item.appendChild(copyLine("Endpoint", row.endpoint)); }
      // Like `llmctl ls`: the stable gateway names that reach this model.
      (row.aliases || []).forEach(function (alias) {
        item.appendChild(copyLine("Gateway name", alias));
      });
      box.appendChild(item);
    });
    if (serving.length && gatewayUrl) { box.appendChild(copyLine("Gateway", gatewayUrl)); }
    servingCount = serving.length;
    $("stop-all-btn").disabled = !serving.length || jobRunning;
  }

  // Shows lines in the confirm panel; with an action, the confirm button runs it.
  function openConfirm(lines, label, action) {
    var body = $("confirm-body");
    var go = $("confirm-run-btn");
    clear(body);
    lines.forEach(function (line) { body.appendChild(el("div", line)); });
    confirmAction = action || null;
    go.textContent = label;
    go.hidden = !action;
    go.disabled = false;
    $("confirm-panel").hidden = false;
  }

  // Hides the confirm panel and forgets the pending action.
  function closeConfirm() {
    confirmAction = null;
    $("confirm-panel").hidden = true;
  }

  // Reports an action failure (409 busy, 404 unknown) in the error area.
  function failAction(err) {
    closeConfirm();
    showError(err.message);
  }

  // Runs the pending action once; the button locks so a double click sends one request.
  function onConfirm() {
    if (!confirmAction || $("confirm-run-btn").disabled) { return; }
    $("confirm-run-btn").disabled = true;
    confirmAction();
  }

  // Dry-runs a swap and asks the user to confirm the preview.
  function startRun(name) {
    apiWithAuth("POST", "/swap/run", { name: name, dry_run: true }).then(function (data) {
      clearError();
      openConfirm(data.preview || [], "Run " + data.name, function () { confirmRun(name); });
    }, failAction);
  }

  // Sends the real run: 202 opens the job panel, 200 means it already serves.
  function confirmRun(name) {
    apiWithAuth("POST", "/swap/run", { name: name }).then(function (data) {
      if (data.job_id) {
        closeConfirm();
        attachJob(data.job_id);
        return;
      }
      var lines = [data.already_serving ? "Already serving: " + data.name : data.name];
      if (data.endpoint) { lines.push("Endpoint: " + data.endpoint); }
      openConfirm(lines, "", null);
      refresh();
    }, failAction);
  }

  // Dry-runs a stop for {name} or {all:true} and lists what it would free.
  function startStop(target) {
    var dry = Object.assign({ dry_run: true }, target);
    apiWithAuth("POST", "/swap/stop", dry).then(function (data) {
      clearError();
      var lines = (data.items || []).map(function (i) {
        return i.label + " — frees cards " + i.cards;
      });
      if (!lines.length) { openConfirm(["Nothing is serving."], "", null); return; }
      openConfirm(lines, "Stop", function () { confirmStop(target); });
    }, failAction);
  }

  // Sends the real stop, lists each result, then refreshes the list.
  function confirmStop(target) {
    apiWithAuth("POST", "/swap/stop", target).then(function (data) {
      var lines = (data.results || []).map(function (r) {
        return (r.stopped ? "Stopped " : "Could not stop ") + r.label;
      });
      openConfirm(lines, "", null);
      refresh();
    }, failAction);
  }

  // Enables or disables every Run, Stop and Stop all button.
  function setJobRunning(flag) {
    jobRunning = flag;
    document.querySelectorAll("td.actions button").forEach(function (b) { b.disabled = flag; });
    $("stop-all-btn").disabled = flag || !servingCount;
  }

  // Draws one job snapshot; returns true once the job has finished.
  function renderJob(job) {
    var running = job.status === "running";
    var text = job.exit_code === 0 ? "Ready" : "Failed (exit " + job.exit_code + ")";
    $("job-status").textContent = job.name + ": " + (running ? job.status : text);
    var log = $("job-log");
    log.textContent = (job.output || []).slice(-200).join("\n");
    log.scrollTop = log.scrollHeight;
    return !running;
  }

  // Fetches the job once; on completion stops polling and refreshes the list.
  function pollJob() {
    var id = jobId;
    if (id === null || jobBusy) { return; }
    jobBusy = true;
    apiWithAuth("GET", "/swap/jobs/" + encodeURIComponent(id)).then(function (job) {
      jobBusy = false;
      if (id === jobId && renderJob(job)) {
        window.clearInterval(jobTimer);
        lastDoneId = id;
        jobId = null;
        setJobRunning(false);
        refresh();
      }
    }, function (err) {
      jobBusy = false;
      showError(err.message);
      if (err.status === 404 && id === jobId) {
        window.clearInterval(jobTimer);
        lastDoneId = id;
        jobId = null;
        setJobRunning(false);
      }
    });
  }

  // Opens the job panel on a job id and follows it until it finishes.
  function attachJob(id) {
    window.clearInterval(jobTimer);
    jobId = id;
    jobBusy = false;
    $("job-panel").hidden = false;
    $("job-status").textContent = "running";
    $("job-log").textContent = "";
    setJobRunning(true);
    pollJob();
    jobTimer = window.setInterval(pollJob, JOB_POLL_MS);
  }

  // Fetches the list once and renders it; clears the error only on success.
  function refresh() {
    if (pollInFlight) { refreshQueued = true; return Promise.resolve(); }
    pollInFlight = true;
    return apiWithAuth("GET", "/swap").then(
      function (data) {
        pollInFlight = false;
        renderRows(data.rows || []);
        renderServing(data.rows || [], data.gateway_url);
        clearError("list");
        if (data.job && jobId === null && data.job.job_id !== lastDoneId) {
          attachJob(data.job.job_id);
        }
        if (refreshQueued) { refreshQueued = false; refresh(); }
      },
      function (err) {
        pollInFlight = false;
        showError(err.message, "list");
        if (refreshQueued) { refreshQueued = false; refresh(); }
      }
    );
  }

  // Restarts the 5 s loop (and refreshes now); a hidden tab makes no requests.
  function startPolling() {
    window.clearInterval(pollTimer);
    if (!document.hidden) {
      refresh();
      pollTimer = window.setInterval(refresh, POLL_MS);
    }
  }

  // Wires static controls and begins polling.
  function init() {
    renderHeader();
    $("token-clear").addEventListener("click", function () {
      setToken("");
      $("token-input").value = "";
    });
    $("refresh-btn").addEventListener("click", refresh);
    $("stop-all-btn").addEventListener("click", function () { startStop({ all: true }); });
    $("confirm-run-btn").addEventListener("click", onConfirm);
    $("confirm-cancel-btn").addEventListener("click", closeConfirm);
    // Pauses polling while the tab is hidden and resumes when it is shown.
    document.addEventListener("visibilitychange", startPolling);
    startPolling();
  }

  init();
})();
