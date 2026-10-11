"""Download the complete published benchmark once and verify the official MD5."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


class BenchmarkDownload:
    """Own the pinned public URL, archive length and checksum."""

    url = 'https://zenodo.org/records/14225758/files/masif-neosurf-benchmark-dataset.zip?download=1'
    size = 9961569979
    md5 = '2b45977254f45c6e229390cc62f3685d'

    def run(self, output):
        """Stream to a partial file, stop on errors, and promote only verified bytes."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        partial, final = output / 'dataset.zip.part', output / 'dataset.zip'
        checksum, count = hashlib.md5(), 0
        with urlopen(self.url, timeout=90) as response, partial.open('wb') as stream:
            if response.status != 200:
                raise ValueError('Expected a complete HTTP200 archive response')
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
                checksum.update(block)
                count += len(block)
                if count % (128 * 1024 * 1024) == 0:
                    print(f'Downloaded {count} / {self.size} bytes', flush=True)
        if count != self.size or checksum.hexdigest() != self.md5:
            raise ValueError('Full archive length or official MD5 mismatch')
        partial.replace(final)
        report = dict(url=self.url, bytes=count, md5=checksum.hexdigest(), verified=True)
        (output / 'download.json').write_text(json.dumps(report, indent=2) + '\n')
        return report


def main():
    """Download into an explicit output folder and print checksum evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(BenchmarkDownload().run(args.output), indent=2))


if __name__ == '__main__':
    main()
