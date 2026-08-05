"""Test package marker.

Makes the shared helpers in this directory (``_tui_isolation``) importable
under every pytest import mode: without it, a relative import works only under
``importlib`` and an absolute one only under ``prepend``, so the suite would
pass locally and fail on a CI whose pytest resolves to the other default.
"""
