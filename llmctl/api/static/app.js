(function () {
  "use strict";

  // Poll cadence; short enough to feel live, long enough to spare the host.
  var POLL_MS = 5000;
  var TOKEN_KEY = "llmctl_token";
  var COLUMNS = ["#", "Model", "Engine", "Name", "Cards", "State", "Last load"];

  var pollTimer = null;
  var pollInFlight = false;
  var lastJobId = null;

  // Shorthand lookup so the rest of the file stays readable.
  function $(id) {
    return document.getElementById(id);
  }

  // ---- T4 hooks -----------------------------------------------------------
  // T4 replaces these; the core only calls them, so the wiring stays in one place.
  var hooks = {
    // Called when a Run button is pressed for a swap target name.
    onRun: function (name) { void name; },
    // Called when a Stop button is pressed; target is a name or "all".
    onStop: function (target) { void target; },
    // Called when the list reports a running job, so T4 can follow its output.
    attachJob: function (id) { void id; }
  };

  // ---- token storage ------------------------------------------------------

  // Reads the saved token; sessionStorage may throw when storage is blocked.
  function getToken() {
    try {
      return window.sessionStorage.getItem(TOKEN_KEY) || "";
    } catch (e) {
      return "";
    }
  }

  // Saves or clears the token for this tab only, never persisting beyond it.
  function setToken(value) {
    try {
      if (value) {
        window.sessionStorage.setItem(TOKEN_KEY, value);
      } else {
        window.sessionStorage.removeItem(TOKEN_KEY);
      }
    } catch (e) {
      /* storage blocked: the token simply will not be remembered */
    }
  }

  // ---- API helper ---------------------------------------------------------

  // Builds an Error that carries the HTTP status so callers can react to 401.
  function apiError(message, status) {
    var err = new Error(message);
    err.status = status;
    return err;
  }

  // Turns a FastAPI error body into one readable string.
  function detailOf(data, fallback) {
    if (data && typeof data.detail === "string") {
      return data.detail;
    }
    if (data && data.detail !== undefined) {
      return JSON.stringify(data.detail);
    }
    return fallback;
  }

  // Single entry point to the server; relative /swap paths keep it same-origin.
  function api(method, path, body) {
    if (path.indexOf("/swap") !== 0) {
      return Promise.reject(apiError("refusing non-/swap path", 0));
    }
    var headers = { Accept: "application/json" };
    var token = getToken();
    if (token) {
      headers["Authorization"] = "Bearer " + token;
    }
    var init = { method: method, headers: headers };
    if (body !== undefined && body !== null) {
      headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    return fetch(path, init).then(
      function (res) {
        return res.text().then(function (text) {
          var data = null;
          if (text) {
            try {
              data = JSON.parse(text);
            } catch (e) {
              data = null;
            }
          }
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

  // ---- error area ---------------------------------------------------------

  // Shows a message in the alert area.
  function showError(message) {
    var area = $("error-area");
    area.textContent = message;
    area.hidden = false;
  }

  // Hides the alert area once things work again.
  function clearError() {
    var area = $("error-area");
    area.textContent = "";
    area.hidden = true;
  }

  // ---- 401 handling -------------------------------------------------------

  // One shared promise, so concurrent 401s all resume on a single submit.
  var tokenPrompt = null;

  // Asks for a token; resolves once the user submits one, so the caller can retry.
  function askForToken() {
    if (tokenPrompt) {
      return tokenPrompt;
    }
    var form = $("token-form");
    form.hidden = false;
    tokenPrompt = new Promise(function (resolve) {
      form.addEventListener("submit", function onSubmit(ev) {
        ev.preventDefault();
        var value = $("token-input").value.trim();
        if (!value) {
          return;
        }
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
      if (err.status !== 401) {
        throw err;
      }
      setToken("");
      showError("Authentication required: enter the API token.");
      return askForToken().then(function () {
        clearError();
        return apiWithAuth(method, path, body);
      });
    });
  }

  // Lets the user drop a stored token from the form.
  function wireTokenClear() {
    $("token-clear").addEventListener("click", function () {
      setToken("");
      $("token-input").value = "";
    });
  }

  // ---- formatting ---------------------------------------------------------

  // Mirrors lanes.format_duration so the UI and CLI agree: 25s, 2m 05s.
  function formatDuration(seconds) {
    if (seconds === null || seconds === undefined) {
      return "—";
    }
    var total = Math.round(seconds);
    var minutes = Math.floor(total / 60);
    var secs = total % 60;
    if (minutes) {
      return minutes + "m " + (secs < 10 ? "0" : "") + secs + "s";
    }
    return secs + "s";
  }

  // The API sends cards as a string or a list; show either as one string.
  function cardsText(cards) {
    if (Array.isArray(cards)) {
      return cards.join(", ");
    }
    if (cards === null || cards === undefined) {
      return "";
    }
    return String(cards);
  }

  // ---- rendering ----------------------------------------------------------

  // Small element factory; textContent only, so server data can never become markup.
  function el(tag, text, className) {
    var node = document.createElement(tag);
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    if (className) {
      node.className = className;
    }
    return node;
  }

  // Copies text to the clipboard and briefly confirms on the button.
  function copyText(text, button) {
    // Shows a status label, then restores "Copy" (fixed, so double-clicks can't stick).
    var done = function (label) {
      button.textContent = label;
      window.setTimeout(function () { button.textContent = "Copy"; }, 1500);
    };
    if (!navigator.clipboard) {
      done("No clipboard");
      return;
    }
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
    var button = el("button", "Copy");
    button.type = "button";
    button.addEventListener("click", function () { copyText(value, button); });
    line.appendChild(button);
    return line;
  }

  // Fills the table header once, since the static HTML header may differ.
  function renderHeader() {
    var head = $("rows-table").tHead;
    while (head.firstChild) {
      head.removeChild(head.firstChild);
    }
    var tr = el("tr");
    COLUMNS.forEach(function (name) { tr.appendChild(el("th", name)); });
    tr.appendChild(el("th", "Action"));
    head.appendChild(tr);
  }

  // Builds the action cell: Run for every row, Stop for serving rows.
  function actionCell(row, serving) {
    var td = el("td", null, "actions");
    var run = el("button", "Run");
    run.type = "button";
    run.addEventListener("click", function () { hooks.onRun(row.name); });
    td.appendChild(run);
    if (serving) {
      var stop = el("button", "Stop");
      stop.type = "button";
      stop.addEventListener("click", function () { hooks.onStop(row.name); });
      td.appendChild(stop);
    }
    return td;
  }

  // Builds the "Name" cell; serving rows add endpoint and alias details below it.
  function nameCell(row, serving, gatewayUrl) {
    var td = el("td");
    td.appendChild(el("div", row.name));
    if (!serving) {
      return td;
    }
    var details = el("div", null, "serving-details");
    if (row.endpoint) {
      details.appendChild(copyLine("Endpoint", row.endpoint));
    }
    (row.aliases || []).forEach(function (alias) {
      var line = copyLine("Alias", alias);
      if (gatewayUrl) {
        line.appendChild(el("span", " via " + gatewayUrl, "alias-gateway"));
      }
      details.appendChild(line);
    });
    td.appendChild(details);
    return td;
  }

  // Redraws the table body from the /swap payload.
  function renderRows(rows, gatewayUrl) {
    var body = $("rows-body");
    while (body.firstChild) {
      body.removeChild(body.firstChild);
    }
    rows.forEach(function (row) {
      var serving = row.state === "serving";
      var tr = el("tr", null, serving ? "serving" : "");
      tr.appendChild(el("td", row.number));
      tr.appendChild(el("td", row.model));
      tr.appendChild(el("td", row.engine));
      tr.appendChild(nameCell(row, serving, gatewayUrl));
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
    while (box.firstChild) {
      box.removeChild(box.firstChild);
    }
    var serving = rows.filter(function (r) { return r.state === "serving"; });
    if (!serving.length) {
      box.appendChild(el("p", "Nothing is serving."));
    }
    serving.forEach(function (row) {
      var item = el("div", null, "serving-item");
      item.appendChild(el("strong", row.name));
      item.appendChild(el("span", " (" + row.model + ", cards " + cardsText(row.cards) + ")"));
      if (row.endpoint) {
        item.appendChild(copyLine("Endpoint", row.endpoint));
      }
      box.appendChild(item);
    });
    if (serving.length && gatewayUrl) {
      box.appendChild(copyLine("Gateway", gatewayUrl));
    }
    $("stop-all-btn").disabled = !serving.length;
  }

  // ---- polling ------------------------------------------------------------

  // Fetches the list once and renders it; clears the error only on success.
  function refresh() {
    if (pollInFlight) {
      return Promise.resolve();
    }
    pollInFlight = true;
    return apiWithAuth("GET", "/swap").then(
      function (data) {
        pollInFlight = false;
        renderRows(data.rows || [], data.gateway_url);
        renderServing(data.rows || [], data.gateway_url);
        clearError();
        var job = data.job;
        if (job && job.job_id !== lastJobId) {
          lastJobId = job.job_id;
          hooks.attachJob(job.job_id);
        }
      },
      function (err) {
        pollInFlight = false;
        showError(err.message);
      }
    );
  }

  // Stops the timer so a hidden tab makes no requests.
  function stopPolling() {
    if (pollTimer !== null) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  // Starts the 5 s loop (and refreshes now) unless the tab is hidden.
  function startPolling() {
    stopPolling();
    if (document.hidden) {
      return;
    }
    refresh();
    pollTimer = window.setInterval(refresh, POLL_MS);
  }

  // Pauses polling while the tab is hidden and resumes when it is shown.
  function onVisibilityChange() {
    if (document.hidden) {
      stopPolling();
    } else {
      startPolling();
    }
  }

  // ---- startup ------------------------------------------------------------

  // Wires static controls and begins polling.
  function init() {
    renderHeader();
    wireTokenClear();
    $("refresh-btn").addEventListener("click", refresh);
    $("stop-all-btn").addEventListener("click", function () { hooks.onStop("all"); });
    document.addEventListener("visibilitychange", onVisibilityChange);
    startPolling();
  }

  // Shared with T4 code appended below, which lives in this same closure.
  var core = {
    api: apiWithAuth,
    refresh: refresh,
    showError: showError,
    clearError: clearError,
    formatDuration: formatDuration,
    hooks: hooks
  };
  void core;

  init();
})();
