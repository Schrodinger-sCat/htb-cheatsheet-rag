"""Unit tests for the chunking logic — the part with the subtlest bugs.

Run with:  python -m pytest tests/ -q     (or: python -m tests.test_ingest)
No Ollama or index required.
"""
from pathlib import Path
import tempfile

from app import ingest


def _chunks_for(md: str):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "htb-example-box.md"
        p.write_text(md, encoding="utf-8")
        return ingest.chunk_file(p)


def test_machine_name_from_filename():
    slug, title = ingest.machine_from_filename(Path("htb-example-box.md"))
    assert slug == "example-box"
    assert title == "Example Box"


def test_headings_inside_code_fences_are_not_headings():
    md = (
        "## Recon\n"
        "Some intro text that is long enough to be kept as a chunk body here.\n"
        "```bash\n"
        "# this is a shell comment, NOT a markdown heading\n"
        "nmap -p- 10.10.10.10\n"
        "```\n"
        "### Privesc\n"
        "We abuse SeImpersonatePrivilege with GodPotato to get a SYSTEM shell here, "
        "which is more than enough characters to survive the minimum-length filter.\n"
    )
    chunks = _chunks_for(md)
    paths = {c.heading_path for c in chunks}
    # The '# this is a shell comment' line must NOT create a heading level.
    assert not any("shell comment" in p for p in paths), paths
    assert any(p.endswith("Privesc") for p in paths), paths
    assert any("Recon" in p for p in paths), paths


def test_metadata_extraction():
    md = (
        "# Intro\n"
        "OS ![Windows](/icons/Windows.webp)\n"
        "Insane\n\n"
        "## Recon\n"
        "This body needs to be comfortably long enough to survive the minimum-length "
        "filter, so we add a second sentence with plenty of characters in it here.\n"
    )
    chunks = _chunks_for(md)
    assert chunks
    # OS/difficulty are file-level metadata, attached to every chunk.
    assert chunks[0].os == "Windows"
    assert chunks[0].difficulty == "Insane"
    assert chunks[0].machine_title == "Example Box"


def test_retrieval_text_has_machine_and_heading():
    md = "## Foothold\n" + ("Kerberoast the SPN with GetUserSPNs and crack it. " * 4) + "\n"
    chunks = _chunks_for(md)
    rt = chunks[0].retrieval_text()
    assert "Example Box" in rt
    assert "Foothold" in rt


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
