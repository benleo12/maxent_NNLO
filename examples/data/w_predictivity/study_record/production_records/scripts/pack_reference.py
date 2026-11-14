"""Pack disjoint reference sums into eight histograms before production."""
from pathlib import Path
import json,hashlib,argparse,re
ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();report={}
assert not (a.root/'runs').exists()
for card in sorted((a.root/'cards').glob('*/*/*/rc.run')):
 text=card.read_text();before=hashlib.sha256(text.encode()).hexdigest();group=card.parts[-3]
 if group=='V':
  assert 'expl_qc_' in text and 'pred_region' not in text
  text='\n'.join(line for line in text.splitlines() if not line.strip().startswith('expl_'))+'\n'
  lines=['  pred_region%d > region%d min=0 max=9 nbins=9'%(k,k) for k in range(1,9)]
  text=text.replace('END_HISTOGRAMS','\n'.join(lines)+'\nEND_HISTOGRAMS')
 else:text=re.sub(r'production\s*=.*','production = 20000[5]',text)
 card.write_text(text);card.with_name('pilot.run').write_text(re.sub(r'production\s*=.*','production = 1000[2]',text))
 report[str(card.relative_to(a.root))]=dict(before=before,after=hashlib.sha256(text.encode()).hexdigest())
(a.root/'booking/packed_reference.json').write_text(json.dumps(report,indent=2)+'\n')
print('Reference packing and100k-point training task size recorded')
