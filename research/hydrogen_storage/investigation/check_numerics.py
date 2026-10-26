"""Check sensitivity results with tighter tolerances and perturbed initial guesses."""
import json
from pathlib import Path
import numpy as np
from model import *
from atmodeller.engine import compute_implied_log_stability

HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs'


def main():
    ref=json.loads((OUT/'study_meb.json').read_text());cases=ref['cases']
    selected=[i for i,c in enumerate(cases) if c['T_K']==3000 and c['H_wt_percent']==1 and c['Z'] in [1,100]
              and c['mode'] in ['h2','water','both'] and c['factor'] in [.1,1.,10.]]
    a=lambda key:np.array([cases[i][key] for i in selected])
    model,mass=make_meb(a('H_wt_percent'),a('Z'),a('T_K'),a('h2_scale'),a('h2o_scale'),tolerance=1e-10)
    guess=np.load(OUT/'study_meb_solution.npy')[selected].copy();ns=guess.shape[1]//2
    guess[:,:ns]+=np.random.default_rng(260126).normal(0,.05,size=guess[:,:ns].shape)
    guess[:,ns:]=np.asarray(compute_implied_log_stability(model.parameters,guess[:,:ns]))
    print('Independent numerical check:',len(selected),'cases, Newton, 1e-10 tolerance, perturbed starts',flush=True)
    out=model.solve(guess);data=extract(out,mass)
    discrepancies={s:float(np.max(abs(x/np.asarray(ref['data']['mole_fraction'][s])[selected]-1))) for s,x in data['mole_fraction'].items()}
    checks=dict(n_cases=len(selected),n_success=int(sum(data['success'])),
                seed=260126,log_moles_perturbation_sigma=.05,tolerance=1e-10,
                maximum_relative_gas_mole_fraction_difference=max(discrepancies.values()),
                differences_by_species=discrepancies,
                maximum_relative_element_error=max(float(np.max(abs(v))) for v in data['relative_element_errors'].values()))
    (OUT/'numerical_robustness.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2),flush=True)
    assert np.all(data['success'])
    assert checks['maximum_relative_gas_mole_fraction_difference']<1e-6


if __name__=='__main__':main()
