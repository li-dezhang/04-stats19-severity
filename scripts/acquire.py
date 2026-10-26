"""Acquire only the two authorized annual collision tables and official dictionary."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,urllib.request
ROOT=Path(__file__).resolve().parents[1]
FILES={
 'collisions_2024.csv':('https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2024.csv',19208488),
 'collisions_2025.csv':('https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2025.csv',19349153),
 'data_guide_2025.xlsx':('https://assets.publishing.service.gov.uk/media/6ab2a71d997a4b2950cced58/dft-road-casualty-statistics-road-safety-open-dataset-data-guide-2025.xlsx',85112)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
 assert shutil.disk_usage(raw).free>1_000_000_000
 manifest_path=ROOT/'data/manifest.json'
 old=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
 prior={r['path']:r for r in old.get('files',[]) if isinstance(r,dict) and 'path' in r}
 rows=[]
 for name,(url,expected) in FILES.items():
  path=raw/name;rel=str(path.relative_to(ROOT))
  if not path.exists():
   with urllib.request.urlopen(url,timeout=60) as response,path.with_suffix(path.suffix+'.part').open('xb') as f:
    size=0
    while b:=response.read(1024*1024):
     size+=len(b)
     if size>30_000_000:raise ValueError('Unexpected size exceeds budget')
     f.write(b)
   tmp=path.with_suffix(path.suffix+'.part')
   if tmp.stat().st_size!=expected:raise ValueError('Source size changed; inspect before continuing')
   tmp.rename(path);path.chmod(0o444)
  row={'path':rel,'url':url,'bytes':path.stat().st_size,'sha256':sha(path)}
  if rel in prior:assert row['sha256']==prior[rel]['sha256'],'Existing raw file changed'
  rows.append(row);print(json.dumps(row),flush=True)
 if not old.get('acquired_utc'):
  manifest={'source_page':'https://www.gov.uk/government/statistical-data-sets/road-safety-open-data','licence':'Open Government Licence v3.0','licence_url':'https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/','attribution':'Contains public sector information licensed under the Open Government Licence v3.0. Source: Department for Transport, STATS19.','acquired_utc':datetime.now(timezone.utc).isoformat(),'files':rows}
  manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
