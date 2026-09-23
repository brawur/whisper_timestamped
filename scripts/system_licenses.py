"""Append Debian notices to the existing pip-licenses report."""
import argparse
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import quote


def collect(packages, doc_root, common_root):
    entries = []
    sources = set()
    for line in packages.splitlines():
        package, version, source, source_version, status = line.split("\t")
        if status != "installed":
            continue
        source = source or package
        source_version = source_version or version
        # Fail the build rather than silently omit a package's notices.
        text = (doc_root / package / "copyright").read_text(encoding="utf-8")
        labels = sorted(set(re.findall(r"^License:\s*(.+)$", text, re.MULTILINE)))
        entries.append({
            "Name": "debian/" + package, "Version": version,
            "License": " / ".join(labels) or "See Debian copyright (component-specific licenses)",
            "LicenseText": text,
            "URL": "https://sources.debian.org/src/" + quote(source, safe="") + "/" + quote(source_version, safe="") + "/",
            "SourcePackage": source, "SourceVersion": source_version,
            "PackageSystem": "deb",
        })
        sources.add((source, source_version))
    for path in sorted(common_root.iterdir()):
        if path.is_file():
            entries.append({
                "Name": "debian/common-licenses/" + path.name,
                "Version": "bundled", "License": path.name,
                "LicenseText": path.read_text(encoding="utf-8"),
                "URL": "https://www.debian.org/legal/licenses/",
                "PackageSystem": "deb",
            })
    return entries, [{"package": p, "version": v} for p, v in sorted(sources)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()
    packages = subprocess.check_output([
        "dpkg-query", "-W",
        "-f=${Package}\t${Version}\t${source:Package}\t${source:Version}\t${db:Status-Status}\n",
    ], text=True)
    entries, sources = collect(packages, Path("/usr/share/doc"), Path("/usr/share/common-licenses"))
    previous = json.loads(args.report.read_text(encoding="utf-8"))
    previous = [entry for entry in previous if entry.get("PackageSystem") != "deb"]
    args.report.write_text(json.dumps(previous + entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.sources.write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
