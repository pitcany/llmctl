"""Top-level package entrypoints."""

from __future__ import annotations

from llmctl.cli import app


def main() -> None:
    """Run the Typer CLI application."""
    app()


if __name__ == "__main__":
    main()
