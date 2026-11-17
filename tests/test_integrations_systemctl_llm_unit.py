"""Engine units go through llm-unit when it is installed (llmctl M4b step 3)."""

from llmctl.integrations.systemctl import SystemctlRunner, SystemctlVerb, llm_unit_for


def _capture():
    seen = []

    class Done:
        returncode, stdout, stderr = 0, "", ""

    def runner(argv):
        seen.append(argv)
        return Done()

    return seen, runner


def test_managed_units_use_llm_unit_and_others_do_not(monkeypatch):
    monkeypatch.setenv("LLM_UNIT_BIN", "/opt/llm-unit")
    seen, runner = _capture()
    r = SystemctlRunner(runner)
    r.run(SystemctlVerb.RESTART, "vllm-tp")
    r.run(SystemctlVerb.STOP, "ollama.service")
    r.run(SystemctlVerb.STOP, "llama-server")
    r.run(SystemctlVerb.STOP, "vllm-engine", user=True)  # explicit user scope stays explicit
    assert seen == [
        ["/opt/llm-unit", "restart", "vllm-tp"],
        ["/opt/llm-unit", "stop", "ollama.service"],
        ["sudo", "systemctl", "stop", "llama-server"],
        ["systemctl", "--user", "stop", "vllm-engine"],
    ]


def test_none_disables_llm_unit(monkeypatch):
    monkeypatch.setenv("LLM_UNIT_BIN", "none")
    assert llm_unit_for("vllm-engine") is None
