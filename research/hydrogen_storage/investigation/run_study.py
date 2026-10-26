"""Controlled MEB storage experiments with independent thermochemical checks.

Run after validate_reference.py. Factors 0.1--10 are exploratory, not confidence intervals.
"""
import json
from pathlib import Path
import time
import sys
import numpy as np
from model import *
from atmodeller.engine import compute_implied_log_stability
from atmodeller.solvers import make_solver_with_jit_batch_only

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"


def scenarios():
    cases = []
    factors = np.logspace(-1, 1, 9)
    for mode in ["h2", "water", "both", "fixed_pressure"]:
        for t in ([2500., 3000., 3500.] if mode == "h2" else [3000.]):
            for z in [1., 10., 100.]:
                for h in [0.1, 0.3, 1.]:
                    for s in factors:
                        cases.append(dict(mode=mode, T_K=t, Z=z, H_wt_percent=h,
                            factor=s, h2_scale=s if mode != "water" else 1.,
                            h2o_scale=s if mode in ["water", "both"] else 1.))
    for z in [1.,10.,100.]:
        for h in [0.1,0.3,1.]:
            for s in [10**-.01,10**.01]:
                cases.append(dict(mode="local_h2",T_K=3000.,Z=z,H_wt_percent=h,
                                  factor=s,h2_scale=s,h2o_scale=1.))
    return cases


def diagnostics(data, cases):
    # Independent net reaction SiO + CH4 -> SiH4 + CO.
    temperature = jnp.asarray([c['T_K'] for c in cases])
    pressure = jnp.asarray(data['pressure_bar'])
    signs = {'H4Si_g':1, 'CO_g':1, 'OSi_g':-1, 'CH4_g':-1}
    gibbs = sum(signs[s.name] * s.get_gibbs_over_RT(temperature, pressure)
                for s in make_gases(False) if s.name in signs)
    ln_k = -np.asarray(gibbs)
    ln_q = sum(sign * np.log(data['fugacity_bar'][s]) for s, sign in signs.items())
    phi = {s: data['fugacity_bar'][s] / data['partial_pressure_bar'][s] for s in signs}
    si_co = np.log(data['fugacity_bar']['OSi_g']/data['fugacity_bar']['CO_g'])
    eos = np.log(phi['CH4_g']/phi['H4Si_g'])
    predicted = ln_k + si_co + eos
    ratio = data['mole_fraction']['H4Si_g']/data['mole_fraction']['CH4_g']
    return dict(ln_K_net=ln_k, ln_Q_minus_ln_K=ln_q-ln_k,
                ln_ratio_from_SiO_CO=si_co, ln_ratio_from_EOS=eos,
                ratio=ratio, ln_ratio_identity_residual=np.log(ratio)-predicted,
                dissolved_H_fraction=data['element_phase_mass_kg']['silicate_melt']['H']/data['mass_constraints_kg']['H'],
                gas_Si_C_mass_ratio=data['element_phase_mass_kg']['gas']['Si']/data['element_phase_mass_kg']['gas']['C'])


def main():
    start = time.time()
    cases = scenarios()
    n = len(cases)
    a = lambda key: np.asarray([c[key] for c in cases])
    model, mass = make_meb(a('H_wt_percent'), a('Z'), a('T_K'),
                          a('h2_scale'), a('h2o_scale'), pressure=np.full(n,np.nan), method="lm")
    model=eqx.tree_at(lambda m:m.solver,model,make_solver_with_jit_batch_only(model.parameters,model.initializer))
    references = np.load(OUT/'reference_meb_solution.npy')
    h_ref = np.logspace(-1,0,50)
    guesses = []
    start_h = []
    start_z = []
    for c in cases:
        i = int(np.argmin(abs(np.log10(h_ref/c['H_wt_percent']))))
        z0 = 1. if c['Z'] <= 10 else 100.
        guesses.append(references[i if z0 == 1 else 50+i])
        start_h.append(h_ref[i])
        start_z.append(z0)
    guess = np.asarray(guesses)
    reference_guesses=guess.copy()
    excluded={}
    start_fraction=0.
    if '--resume' in sys.argv:
        checkpoint=json.loads((OUT/'continuation_checkpoint.json').read_text())
        start_fraction=checkpoint['fraction']
        excluded={int(k):v for k,v in checkpoint['excluded'].items()}
        guess=np.load(OUT/'study_meb_continuation_latest.npy')
        for i in excluded:guess[i]=reference_guesses[i]
    # Direct large jumps can give non-finite Newton steps. Follow a smooth continuation
    # from a converged archived case instead of accepting failed roots or clipping physics.
    print(f'Solving {n} MEB cases by 100-step continuation',flush=True)
    for fraction in np.linspace(0,1,101):
        if fraction < start_fraction-1e-10:continue
        ff=np.full(n,fraction);ff[list(excluded)]=0.
        hh=np.asarray(start_h)*(a('H_wt_percent')/start_h)**ff
        zz=np.asarray(start_z)*(a('Z')/start_z)**ff
        active=model.update_constraints(mass_constraints=budgets(hh,zz)).update_state(
            temperature=3000.+ff*(a('T_K')-3000.))
        scaled_state=jax.tree_util.tree_map(
            lambda law: eqx.tree_at(lambda x:x.multiplier,law,law.multiplier**(ff if law.multiplier.ndim else fraction))
                        if isinstance(law,ScaledSolubility) else law,
            active.parameters.state,
            is_leaf=lambda x:isinstance(x,ScaledSolubility))
        active=eqx.tree_at(lambda m:m.parameters.state,active,scaled_state)
        output=active.solve(guess)
        guess=np.asarray(output.solution).copy()
        success=np.asarray(output.multi_attempt_solution.success)
        print('Continuation',round(float(fraction),2),'success',int(sum(success)),'/',n,flush=True)
        for i in np.where(~success)[0]:
            excluded[int(i)]=dict(stage='MEB continuation',fraction=float(fraction),case=cases[i],reason='Failed convergence; excluded, not interpreted physically')
            guess[i]=reference_guesses[i]
            print('Excluding',int(i),cases[i],flush=True)
        np.save(OUT/'study_meb_continuation_latest.npy',guess)
        (OUT/'continuation_checkpoint.json').write_text(json.dumps(serializable(dict(fraction=float(fraction),excluded=excluded)),indent=2))
    # Also freeze failures first encountered on the final continuation step before
    # reusing the batch for the pressure controls. Excluded rows never enter results.
    if not np.all(success):
        ff=np.ones(n);ff[list(excluded)]=0.
        hh=np.asarray(start_h)*(a('H_wt_percent')/start_h)**ff
        zz=np.asarray(start_z)*(a('Z')/start_z)**ff
        active=model.update_constraints(mass_constraints=budgets(hh,zz)).update_state(temperature=3000.+ff*(a('T_K')-3000.))
        state=jax.tree_util.tree_map(lambda law:eqx.tree_at(lambda x:x.multiplier,law,law.multiplier**(ff if law.multiplier.ndim else 1.))
            if isinstance(law,ScaledSolubility) else law,active.parameters.state,is_leaf=lambda x:isinstance(x,ScaledSolubility))
        active=eqx.tree_at(lambda m:m.parameters.state,active,state)
        output=active.solve(guess);guess=np.asarray(output.solution).copy()
        assert np.all(np.asarray(output.multi_attempt_solution.success))
    np.save(OUT/'study_meb_solution_initial.npy',np.asarray(output.solution))
    data = extract(output,mass)
    print('Initial success',int(sum(data['success'])),'/',n,flush=True)
    (OUT/'study_meb_initial.json').write_text(json.dumps(serializable({'cases':cases,'data':data}),indent=2))
    assert all(data['success'][i] for i in range(n) if i not in excluded)
    baseline_pressure = {(c['Z'],c['H_wt_percent']):data['pressure_bar'][i] for i,c in enumerate(cases)
                         if c['mode']=='h2' and c['T_K']==3000 and c['factor']==1}
    pressures = np.asarray([baseline_pressure[c['Z'],c['H_wt_percent']] if c['mode']=='fixed_pressure' else np.nan for c in cases])
    print('Solving fixed-pressure controls',flush=True)
    starting_pressure=data['pressure_bar'].copy()
    initial_solution=np.asarray(output.solution).copy()
    base_final_model=active
    fixed_excluded=set()
    for fraction in np.linspace(0,1,41):
        pressure_step=np.where(np.isfinite(pressures),starting_pressure*(pressures/starting_pressure)**fraction,np.nan)
        for i in fixed_excluded:pressure_step[i]=starting_pressure[i]
        fixed_model = base_final_model.update_state(pressure=pressure_step)
        output = fixed_model.solve(guess)
        success=np.asarray(output.multi_attempt_solution.success)
        guess=np.asarray(output.solution).copy()
        for i in np.where(~success)[0]:
            fixed_excluded.add(int(i));excluded[int(i)]=dict(stage='Fixed pressure',fraction=float(fraction),case=cases[i],reason='Failed convergence; excluded')
            guess[i]=initial_solution[i]
        if round(fraction*40)%10==0:print('Fixed-pressure continuation',fraction,'success',int(sum(success)),flush=True)
    keep=np.array([i for i in range(n) if i not in excluded])
    np.save(OUT/'study_meb_solution.npy',np.asarray(output.solution)[keep])
    data = extract(output,mass)
    diag = diagnostics(data,cases)
    def subset(d):return {k:subset(v) if isinstance(v,dict) else np.asarray(v)[keep] for k,v in d.items()}
    data=subset(data);diag=subset(diag);cases=[cases[i] for i in keep]
    checks = dict(n_attempted=n,n_cases=len(keep),n_excluded=len(excluded),n_success=int(sum(data['success'])),
                  maximum_relative_element_error=max(float(np.max(abs(v))) for v in data['relative_element_errors'].values()),
                  maximum_abs_ln_reaction_residual=float(np.max(abs(diag['ln_Q_minus_ln_K']))),
                  maximum_abs_ln_ratio_identity_residual=float(np.max(abs(diag['ln_ratio_identity_residual']))),
                  seconds=time.time()-start)
    payload=dict(cases=cases,data=data,diagnostics=diag,checks=checks,excluded_cases=excluded)
    (OUT/'study_meb.json').write_text(json.dumps(serializable(payload),indent=2))
    print(json.dumps(checks,indent=2),flush=True)
    assert np.all(data['success'])
    assert checks['maximum_relative_element_error'] < 1e-7
    assert checks['maximum_abs_ln_reaction_residual'] < 1e-6
    for i,c in enumerate(cases):
        if c['H_wt_percent']==1 and c['Z']==1 and c['T_K']==3000 and c['factor'] in [.1,1.,10.]:
            print(c['mode'],c['factor'],'P',data['pressure_bar'][i],'Hdiss',diag['dissolved_H_fraction'][i],
                  'SiH4/CH4',diag['ratio'][i],flush=True)


if __name__=='__main__':
    main()
