import ast
import json
from pathlib import Path
from typing import Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".git", "__pycache__", ".venv_sidepython"}
STD_LIB_ALLOW = {
    "argparse",
    "ast",
    "collections",
    "contextlib",
    "csv",
    "hashlib",
    "io",
    "json",
    "logging",
    "math",
    "os",
    "pathlib",
    "pickle",
    "random",
    "re",
    "subprocess",
    "sys",
    "textwrap",
    "time",
    "typing",
    "warnings",
}


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return f.readline().strip() == "version https://git-lfs.github.com/spec/v1"
    except Exception:
        return False


def iter_repo_files() -> list[Path]:
    files: list[Path] = []
    for p in REPO_ROOT.rglob("*"):
        if p.is_dir() and p.name in EXCLUDE_DIRS:
            continue
        if p.is_file() and not any(part in EXCLUDE_DIRS for part in p.parts):
            files.append(p)
    return sorted(files)


def parse_imports(py_path: Path) -> Tuple[set[str], Optional[str]]:
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        return set(), f"{type(exc).__name__}: {exc}"

    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return imports, None


def main() -> None:
    files = iter_repo_files()
    py_files = [p for p in files if p.suffix == ".py"]
    lfs_files = [str(p.relative_to(REPO_ROOT)) for p in files if p.suffix in {".jsonl", ".bin", ".csv"} and is_lfs_pointer(p)]

    missing_by_script: dict[str, list[str]] = {}
    parse_errors: dict[str, str] = {}
    all_external: set[str] = set()

    for py in py_files:
        imports, parse_err = parse_imports(py)
        if parse_err:
            parse_errors[str(py.relative_to(REPO_ROOT))] = parse_err
            continue

        external = sorted(i for i in imports if i not in STD_LIB_ALLOW and i not in {"common"})
        all_external.update(external)
        missing = []
        for mod in external:
            try:
                __import__(mod)
            except Exception:
                missing.append(mod)
        if missing:
            missing_by_script[str(py.relative_to(REPO_ROOT))] = missing

    report = {
        "repo_root": str(REPO_ROOT),
        "total_files": len(files),
        "python_files": len(py_files),
        "lfs_pointer_files": lfs_files,
        "parse_errors": parse_errors,
        "missing_modules_by_script": missing_by_script,
        "all_external_modules_seen": sorted(all_external),
    }

    out = REPO_ROOT / "pipeline" / "audit_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote audit report: {out}")
    print(f"Total files: {report['total_files']}")
    print(f"Python files: {report['python_files']}")
    print(f"LFS pointers: {len(report['lfs_pointer_files'])}")
    print(f"Scripts with missing modules: {len(report['missing_modules_by_script'])}")


if __name__ == "__main__":
    main()
