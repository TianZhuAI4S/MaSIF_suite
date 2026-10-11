"""Fetch or locally verify 43 checksum-pinned public real-protein fixture members.

Only selected ZIP ranges are used; the whole 10 GB archive MD5 is not verified.
HTTP errors (including rate limits) stop immediately without automatic retries.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen
import zlib


def read_range(url, start, stop):
    """Read one inclusive byte range and require an exact HTTP 206 response."""
    request = Request(url, headers={"Range": f"bytes={start}-{stop}",
                                    "User-Agent": "MaSIF numerical fixture reproduction"})
    with urlopen(request, timeout=90) as response:
        expected = f"bytes {start}-{stop}/"
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
            raise ValueError("Server did not honor the exact requested byte range")
        data = response.read()
    if len(data) != stop - start + 1:
        raise ValueError("Incomplete HTTP range")
    return data


def check_member(data, member, sha256=None):
    """Check uncompressed size, central-directory CRC32 and any recorded SHA256."""
    if len(data) != member["size"] or zlib.crc32(data) != member["crc32"]:
        raise ValueError(f"ZIP size/CRC mismatch: {member['path']}")
    digest = hashlib.sha256(data).hexdigest()
    if sha256 is not None and digest != sha256:
        raise ValueError(f"SHA256 mismatch: {member['path']}")
    return digest


def fetch_member(url, member):
    """Retrieve a local ZIP header and its payload at the recovered physical offset."""
    offset = member["offset"]
    header = read_range(url, offset, offset + 29)
    signature, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack("<I5H3I2H", header)
    if signature != 0x04034B50 or flags & 1:
        raise ValueError("Invalid or encrypted local ZIP member")
    end = offset + 30 + name_len + extra_len + member["compressed_size"] - 1
    remainder = read_range(url, offset + 30, end)
    encoding = "utf-8" if flags & (1 << 11) else "cp437"
    if remainder[:name_len].decode(encoding) != member["path"]:
        raise ValueError("Recovered ZIP offset points to a different member")
    payload = remainder[name_len + extra_len:]
    if method == 8:
        data = zlib.decompress(payload, -15)
    elif method == 0:
        data = payload
    else:
        raise ValueError(f"Unsupported ZIP compression: {method}")
    check_member(data, member)
    return data


def main():
    """Verify an explicit output directory; --download enables missing-member retrieval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    specification = Path(__file__).resolve().parents[1] / "tests/fixtures/design_members.json"
    manifest = json.loads(specification.read_text())
    verified, missing = 0, []
    for member in manifest["files"]:
        relative = Path(member["path"]).relative_to("masif-neosurf-benchmark-dataset")
        if ".." in relative.parts:
            raise ValueError("Fixture path escapes output directory")
        path = args.output / relative
        if not path.exists():
            if not args.download:
                missing.append(str(relative))
                continue
            data = fetch_member(manifest["source_url"], member)
            check_member(data, member, member["sha256"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        check_member(path.read_bytes(), member, member["sha256"])
        verified += 1
    print(json.dumps(dict(verified=verified, missing=missing, full_archive_md5_verified=False), indent=2))
    if missing:
        raise SystemExit("Missing fixtures: use --download or supply the official extracted dataset")
    manifest_path = args.output / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
