"""One-time move of diagnostic sums before scientific production starts."""
from pathlib import Path
import argparse,re,json,hashlib
from make_booking import definitions
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();report={}
assert not (a.root/'runs').exists(),'Cannot change cards after production starts'
for card in sorted((a.root/'cards').glob('*/*/*/rc.run')):
 text=card.read_text();group=card.parts[-3]
 assert ('expl_qc_' in text)==(group=='T'),'Unexpected existing stream layout'
 before=hashlib.sha256(text.encode()).hexdigest()
 if group=='T':text='\n'.join(line for line in text.splitlines() if not line.strip().startswith('expl_'))+'\n'
 else:
  lines=['  %s > %s min=-1.001 max=1.001 nbins=1 profile=T'%(r['name'],r['name']) for r in definitions() if r['role']=='exploratory']
  text=text.replace('END_HISTOGRAMS','\n'.join(lines)+'\nEND_HISTOGRAMS')
 card.write_text(text);card.with_name('pilot.run').write_text(re.sub(r'production\s*=.*','production = 1000[2]',text))
 report[str(card.relative_to(a.root))]=dict(before=before,after=hashlib.sha256(text.encode()).hexdigest())
(a.root/'booking/stream_allocation.json').write_text(json.dumps(report,indent=2)+'\n')
print('Moved exploratory sums to independent reference; compiled functions unchanged')
