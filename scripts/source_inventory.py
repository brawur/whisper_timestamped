"""Resolve version-specific source references; never equate metadata with compliance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
from urllib.parse import quote
from urllib.request import Request, urlopen


def fetch_json(url):
    request = Request(url, headers={"User-Agent": "MCC-source-inventory/1.0"})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def same_version(left, right):
    # PyPI may omit insignificant trailing zeros (e.g. 13.0.3.0 -> 13.0.3).
    # Never strip local/pre-release suffixes or guess a different release.
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", left) and re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", right):
        def parts(value):
            values = [int(part) for part in value.split(".")]
            while len(values) > 1 and values[-1] == 0:
                values.pop()
            return values
        return parts(left) == parts(right)
    return left == right


def resolve(entry, fetch=fetch_json):
    entry = dict(entry, artifacts=[], status="review-required")
    try:
        data = fetch(entry["metadata_url"])
        if entry["ecosystem"] == "debian":
            if data.get("package") != entry["source_package"] or data.get("version") != entry["source_version"]:
                raise ValueError("Source metadata does not match installed version")
            for artifact in data["result"]:
                digest = artifact["hash"]
                if len(digest) != 40 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("Invalid snapshot source hash")
                entry["artifacts"].append({"url": "https://snapshot.debian.org/file/" + digest,
                                           "snapshot_sha1": digest})
        else:
            if not same_version(data["info"]["version"], entry["version"]):
                raise ValueError("Source metadata does not match installed version")
            entry["artifacts"] = [
                {"url": item["url"], "filename": item["filename"],
                 "sha256": item["digests"]["sha256"]}
                for item in data["urls"] if item["packagetype"] == "sdist"
            ]
        if entry["artifacts"]:
            entry["status"] = "metadata-resolved"
        else:
            entry["reason"] = "No source archive published in this package index; review upstream sources/license."
    except Exception as error:
        entry["artifacts"] = []
        entry["reason"] = str(error)
    return entry


def inventory(report, fetch=fetch_json):
    entries = []
    for item in report:
        name, version = item["Name"], item["Version"]
        if name.startswith("debian/common-licenses/"):
            continue
        if item.get("PackageSystem") == "deb":
            source, source_version = item["SourcePackage"], item["SourceVersion"]
            entries.append({"name": name, "version": version, "license": item["License"],
                            "ecosystem": "debian", "source_package": source, "source_version": source_version,
                            "metadata_url": "https://snapshot.debian.org/mr/package/" + quote(source, safe="") + "/" + quote(source_version, safe="") + "/srcfiles"})
        elif name != "whisper-timestamped-service":
            entries.append({"name": name, "version": version, "license": item["License"],
                            "ecosystem": "pypi", "metadata_url": "https://pypi.org/pypi/" + quote(name, safe="") + "/" + quote(version, safe="") + "/json"})
    # Many binary Debian packages share one source. Resolve each endpoint once.
    representatives = {entry["metadata_url"]: entry for entry in entries}
    with ThreadPoolExecutor(max_workers=4) as pool:
        resolved = dict(zip(representatives, pool.map(lambda e: resolve(e, fetch), representatives.values())))
    for entry in entries:
        result = resolved[entry["metadata_url"]]
        entry.update({key: result[key] for key in ("artifacts", "status", "reason") if key in result})
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-revision", default="")
    args = parser.parse_args()
    raw = args.report.read_bytes()
    entries = inventory(json.loads(raw))
    revision = args.worker_revision
    worker = {"name": "whisper-timestamped-service", "revision": revision,
              "url": "https://github.com/brawur/whisper_timestamped" + ("/tree/" + quote(revision, safe="") if revision else ""),
              "status": "review-required",
              "reason": "Verify public mirror contains this exact revision and all build files."}
    # Official Python image installs CPython outside dpkg/pip.
    python = {"name": "CPython", "version": platform.python_version(),
              "url": "https://www.python.org/ftp/python/" + platform.python_version() + "/Python-" + platform.python_version() + ".tar.xz",
              "status": "review-required", "reason": "Check base-image provenance, build recipe and patches against this upstream source."}
    output = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
              "license_report_sha256": hashlib.sha256(raw).hexdigest(),
              "verification_scope": "metadata-resolved means exact-version source metadata was retrieved. Archives were not downloaded or checked for completeness; this is not a license compliance approval. Bundled wheel libraries and base-image additions require release review.",
              "worker": worker, "base_runtime": python, "packages": entries}
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    pending = sum(e["status"] == "review-required" for e in entries)
    print(f"Source inventory: {len(entries)} packages, {pending} require manual source review; worker/base-image review also required.")


if __name__ == "__main__":
    main()
