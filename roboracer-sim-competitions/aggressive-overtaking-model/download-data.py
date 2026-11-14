"""Download the pinned complete archive; verify before extracting."""
import hashlib,json,shutil,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 for a in json.loads((ROOT/'DATA_RELEASE.json').read_text())['assets']:
  archive=ROOT/a['name']
  if not archive.exists() or digest(archive)!=a['sha256']:
   tmp=archive.with_suffix('.download')
   with urllib.request.urlopen(a['url']) as r,tmp.open('wb') as f:shutil.copyfileobj(r,f)
   if tmp.stat().st_size!=a['bytes'] or digest(tmp)!=a['sha256']:raise RuntimeError('Archive checksum mismatch; no extraction performed')
   tmp.replace(archive)
  with zipfile.ZipFile(archive) as z:
   index=json.loads(z.read('DATA_ARCHIVE_INDEX.json'))
   for record in index['files']:
    dest=(ROOT/record['path']).resolve()
    if not dest.is_relative_to(ROOT):raise RuntimeError('Invalid archive path')
    b=z.read(record['path'])
    if hashlib.sha256(b).hexdigest()!=record['exported_sha256']:raise RuntimeError('Member checksum mismatch')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
  print('Verified and extracted',len(index['files']),'files.')
if __name__=='__main__':main()
