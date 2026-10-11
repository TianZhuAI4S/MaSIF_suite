"""Verify and extract the official benchmark despite its wrapped ZIP32 offsets."""

import argparse
import json
from pathlib import Path
import struct
import zlib
import zipfile


class BenchmarkArchive:
    """Read central-directory inventory members using verified physical offsets."""

    def __init__(self, archive, inventory=None):
        """Retain local paths and independently recovered physical offsets."""
        self.archive = Path(archive)
        self.members = json.loads(Path(inventory).read_text()) if inventory else self.inventory()

    def inventory(self):
        """Recover physical offsets from ordered central-directory ZIP32 records."""
        result, epoch, previous = [], 0, -1
        with zipfile.ZipFile(self.archive) as archive:
            for member in archive.infolist():
                wrapped = member.header_offset % (1 << 32)
                if wrapped < previous:
                    epoch += 1 << 32
                previous = wrapped
                result.append(dict(path=member.filename, size=member.file_size,
                                   compressed_size=member.compress_size, crc32=member.CRC,
                                   offset=epoch + wrapped))
        return result

    def extract(self, output):
        """Check each header, decoded filename, payload size and CRC32."""
        output = Path(output)
        count, size = 0, 0
        with self.archive.open('rb') as stream:
            for member in self.members:
                relative = Path(member['path']).relative_to('masif-neosurf-benchmark-dataset')
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Member escapes output directory')
                if member['path'].endswith('/'):
                    (output / relative).mkdir(parents=True, exist_ok=True)
                    continue
                stream.seek(member['offset'])
                signature, _, flags, method, _, _, _, _, _, name_len, extra_len = struct.unpack('<I5H3I2H', stream.read(30))
                if signature != 0x04034B50 or flags & 1:
                    raise ValueError('Invalid or encrypted member header')
                name = stream.read(name_len).decode('utf-8' if flags & (1 << 11) else 'cp437')
                if name != member['path']:
                    raise ValueError('Recovered offset points to a different member')
                stream.read(extra_len)
                payload = stream.read(member['compressed_size'])
                if method == 8:
                    data = zlib.decompress(payload, -15)
                elif method == 0:
                    data = payload
                else:
                    raise ValueError(f'Unsupported compression {method}')
                if len(data) != member['size'] or zlib.crc32(data) != member['crc32']:
                    raise ValueError(f'Size/CRC mismatch: {name}')
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                count += 1
                size += len(data)
                if count % 100 == 0:
                    print(f'Extracted {count} files, {size} bytes', flush=True)
        report = dict(files=count, uncompressed_bytes=size, verified_crc32=True,
                      inventory_members=len(self.members), archive=str(self.archive.resolve()))
        (output.parent / 'extraction.json').write_text(json.dumps(report, indent=2) + '\n')
        return report


def main():
    """Extract a local archive and print the verification summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, help='Optional previously verified offset inventory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(BenchmarkArchive(args.archive, args.inventory).extract(args.output), indent=2))


if __name__ == '__main__':
    main()
