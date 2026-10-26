"""Build an offline research report from accepted numerical results only."""
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs'


def main():
    analysis=json.loads((OUT/'analysis_summary.json').read_text())
    meb=json.loads((OUT/'study_meb.json').read_text())
    reference=json.loads((OUT/'reference_meb_validation.json').read_text())
    reference_atm=json.loads((OUT/'reference_atmosphere_validation.json').read_text())
    robustness=json.loads((OUT/'numerical_robustness.json').read_text())
    rows=analysis['selected_cases']
    def row(z,s):return next(r for r in rows if r['mode']=='h2' and r['Z']==z and r['factor']==s)
    low,base,high=[row(1,s) for s in [.1,1.,10.]]
    def number(x):return f'{x:.3g}'
    def figure(name,caption):
        encoded=base64.b64encode((HERE/'figures'/f'{name}.png').read_bytes()).decode()
        return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{encoded}"><figcaption>{caption}</figcaption></figure>'
    selected=[row(z,s) for z in [1,100] for s in [.1,1.,10.]]
    table=''.join(f"<tr><td>{r['Z']:g}×</td><td>{r['factor']:g}</td><td>{100*r['H_dissolved_fraction']:.1f}%</td><td>{number(r['ratio_MEB'])}</td><td>{number(r['ratio_10mbar'])}</td><td>{number(r['xSiH4_10mbar'])}</td><td>{number(r['xCH4_10mbar'])}</td><td>{r['buffer']}</td></tr>" for r in selected)
    checks=analysis['checks'];buffers=analysis['buffer_checks']
    stamp=datetime.now(timezone.utc).isoformat()
    # These stored calculations retain their original engine provenance.
    commit=json.loads((OUT/'provenance.json').read_text())['atmodeller_commit']
    for item in json.loads((HERE/'sources'/'downloaded.json').read_text()):
        assert hashlib.sha256((HERE/'sources'/item['name']).read_bytes()).hexdigest()==item['sha256'],item['name']
    text=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hydrogen storage, condensation, and SiH4/CH4 — Atmodeller investigation</title>
<style>body{{font:17px/1.65 system-ui,sans-serif;color:#172d3c;background:#fcfcfa;margin:0}}main{{max-width:1020px;margin:40px auto;padding:0 24px 60px}}h1{{font-size:35px;line-height:1.18;max-width:850px}}h2{{font-size:24px;margin-top:42px}}h3{{font-size:19px}}a{{color:#006a98}}.meta,figcaption{{font-size:13px;color:#516570}}.lead{{font-size:21px;max-width:920px}}.scope{{border-left:4px solid #d18b35;padding:12px 20px;background:#fff7e9}}.equation{{padding:18px;background:#edf4f7;font-size:20px;text-align:center;margin:18px 0}}figure{{margin:30px 0}}img{{max-width:100%;height:auto}}.tablewrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px 10px;border-bottom:1px solid #d9e1e5;text-align:right;white-space:nowrap}}th{{background:#edf4f7}}th:first-child,td:first-child{{text-align:left}}code,pre{{font:13px/1.6 ui-monospace,monospace}}pre{{background:#edf4f7;padding:18px;overflow:auto}}li{{margin:8px 0}}.small{{font-size:14px}}@media(max-width:600px){{main{{padding:0 16px}}h1{{font-size:28px}}.lead{{font-size:18px}}}}</style></head><body><main>
<p class="meta">Numerical investigation · 26 October 2026 · Atmodeller v2.0.0 · {commit[:12]}</p>
<h1>Can interior hydrogen storage change atmospheric silane relative to methane?</h1>
<p class="lead"><strong>Yes, but condensation can either erase the change or make it much larger.</strong> In the solar-composition example, a {low['ratio_MEB']/high['ratio_MEB']:.1f}-fold change in deep SiH₄/CH₄ becomes an unchanged atmospheric ratio of 0.006382. At 100× solar, the same storage perturbation changes the stable condensates and moves the gas between very different chemical regimes.</p>
<p class="scope"><strong>Scope:</strong> actual Atmodeller calculations, with fixed total elemental budgets and hypothetical multipliers on H₂/H₂O dissolution laws. This is a sensitivity experiment relevant to the proposed demixing project. It does not calculate a spinodal, a phase-separation rate, or the effect of a calibrated demixing free energy.</p>

<h2>What was calculated</h2>
<p>The reference is the TOI-421b setup and archived calculations associated with <a href="https://doi.org/10.1093/mnras/stag133">Hakim et al. (2026)</a>. I used their <a href="https://osf.io/ac6bw/">OSF notebook and numerical tables</a>, then rebuilt the chemical system with the current <a href="https://github.com/ExPlanetology/atmodeller/tree/{commit}">pinned Atmodeller source</a>. The scientific question is whether atmospheric SiH₄/CH₄ remains sensitive to altered hydrogen storage once cooling and condensation are included.</p>
<p>The central planet has mass 6.7 Earth masses, magma-boundary radius 1.65 Earth radii, a fully molten background mantle, a boundary temperature of 3000 K, and total elemental H equal to 1% of planetary mass. Eleven gases track H–He–C–N–O–Si. Liquid SiO₂ supplies the reactive rock component; seven gas dissolution laws couple the gas to a fixed background melt. Atmospheric parcels inherit only the computed gas elemental inventory, then equilibrate at imposed temperature and pressure.</p>
<p>For each H₂ multiplier s, the dissolved concentration law becomes c<sub>H₂</sub> = s c<sub>H₂,reference</sub>, with s from 0.1 to 10. Independent controls scale the H₂O law or both laws. A pressure control holds the deep pressure at its unperturbed value. Other scenarios span H inventories of 0.1, 0.3 and 1 wt%, accreted metallicities of 1, 10 and 100× solar, and deep temperatures of 2500, 3000 and 3500 K.</p>
<p>There are <strong>501 accepted deep scenarios out of 504 attempted</strong>, <strong>615 local atmospheric calculations with condensates</strong>, and <strong>297 gas-only controls</strong>. Every accepted interior case is evaluated at 920 K and 10 mbar. Selected pressure profiles span 1 mbar–100 bar, and temperature controls use 700 and 1200 K. Scenario counts include repeated control baselines; continuation steps are not counted as additional results.</p>

<h2>The interior signal is strong, but its atmospheric expression depends on phases</h2>
<div class="tablewrap"><table><thead><tr><th>Accreted Z</th><th>H₂ law ×</th><th>H dissolved</th><th>Deep SiH₄/CH₄</th><th>Atmospheric ratio</th><th>x(SiH₄)</th><th>x(CH₄)</th><th>Ratio buffer</th></tr></thead><tbody>{table}</tbody></table></div>
<p class="small">Deep temperature 3000 K; 1 wt% H; atmospheric temperature 920 K and pressure 10 mbar. x denotes gas mole fraction. “Other” means the Si+SiC and C+SiC pairs do not coexist; it does not mean no condensates. These hypothetical multipliers are not uncertainty bounds inferred from experiments.</p>
<p>At solar composition, increasing H₂ solubility across the full range raises hydrogen retention from {100*low['H_dissolved_fraction']:.1f}% to {100*high['H_dissolved_fraction']:.1f}% and lowers the deep ratio from {low['ratio_MEB']:.1f} to {high['ratio_MEB']:.2f}. Yet the atmospheric ratio stays fixed. Absolute abundances retain information: atmospheric SiH₄ falls from {number(low['xSiH4_10mbar'])} to {number(high['xSiH4_10mbar'])}, a factor of {low['xSiH4_10mbar']/high['xSiH4_10mbar']:.1f}, and methane changes by the same factor. A constant ratio therefore does not imply an unchanged atmosphere.</p>
{figure('storage_sensitivity','Figure 1. Storage, deep chemistry and cooled atmospheric gas. The flat atmospheric-ratio panel is shown on a physically meaningful scale: fluctuations at floating-point precision are not physical changes. Lines connect sampled scenarios, not a calibrated continuous model of demixing.')}
<p>At 100× solar, weak H₂ storage leaves both free silicon and silicon carbide. At the reference storage law, graphite replaces free silicon and the atmospheric ratio is about 28 million times lower. At the strongest storage perturbation, silicon carbide also disappears and silane becomes extremely scarce. The smallest trace fractions are formal equilibrium outputs; no detectability threshold, reaction timescale or spectrum was calculated.</p>

<h2>Why the ratio becomes insensitive</h2>
<p>When Si and SiC coexist as pure condensed phases, the following balanced reaction fixes the gas ratio:</p>
<div class="equation">2 Si(s) + CH₄(g) ⇌ SiC(s) + SiH₄(g)</div>
<p>With unit condensed-phase activities and ideal atmospheric gases, x(SiH₄)/x(CH₄) = K(T). Pressure and H₂ fugacity cancel. Computing K directly from the packaged standard Gibbs energies gives <strong>0.0063820347 at 920 K</strong>. Additional hydrogen storage changes the amounts of condensates and gas species, while this relation holds until one member of the buffering pair is exhausted.</p>
<p>The alternative carbon-rich pair obeys:</p>
<div class="equation">SiC(s) + CH₄(g) ⇌ 2 C(s) + SiH₄(g)</div>
<p>Its equilibrium constant gives <strong>2.25592 × 10⁻¹⁰ at 920 K and 10 mbar</strong>. The implemented graphite volume correction gives a small pressure dependence for this second buffer. These identities explain the computed plateaus without fitting the output. Of the 615 accepted atmospheric scenarios, {buffers['silicon_and_silicon_carbide_cases']} contain Si+SiC and {buffers['carbon_and_silicon_carbide_cases']} contain C+SiC above a 10⁻¹² kg per kg parcel mass cutoff. Direct Gibbs-energy predictions match their ratios within {max(buffers['maximum_relative_Si_SiC_buffer_error'],buffers['maximum_relative_C_SiC_buffer_error']):.2g} relative error. The remaining {buffers['other_cases']} cases require their actual phase assemblages.</p>
<p>Temperature is consequential even on a plateau: the Si+SiC ratio is 0.001339 at 700 K, 0.006382 at 920 K and 0.021270 at 1200 K. At the reference H₂ law, finite-difference atmospheric sensitivities to H₂ storage are numerically zero for all nine tested combinations of metallicity and H inventory, although the deep sensitivities are nonzero. Large perturbations can cross phase boundaries and invalidate that local insensitivity.</p>
{figure('phase_buffers','Figure 2. Condensate buffering and its temperature dependence. The overlapping 1× and 10× solar curves are intentional. The 100× solar transition is sampled on a 0.25-dex multiplier grid; the connecting segments do not establish an exact transition threshold.')}

<h2>Controls establish what causes the result</h2>
<p>At the magma boundary, the balanced net reaction SiO + CH₄ ⇌ SiH₄ + CO implies</p>
<div class="equation">x(SiH₄) / x(CH₄) = K<sub>net</sub> · [f(SiO) / f(CO)] · [φ(CH₄) / φ(SiH₄)].</div>
<p>Thus hydrogen acts through the coupled elemental redistribution and real-gas state; there is no independent explicit H₂ factor in this ratio identity. For the central solar case, going from s = 1 to s = 10 changes log₁₀ of the ratio by −1.434: −1.266 comes from f(SiO)/f(CO), and −0.168 from the fugacity-coefficient ratio. All 501 accepted deep runs satisfy the direct reaction check to better than 10⁻¹² in ln(Q/K).</p>
<p>Pressure feedback matters. The full endpoint change is 69-fold with self-consistent mechanical pressure, versus about 324-fold if pressure is held at the reference value. H₂O-law and joint-law controls also produce different, sometimes nonmonotonic deep responses. The fraction of total hydrogen stored is therefore insufficient by itself to specify the resulting chemistry.</p>
<p>The atmospheric gas-only control disables condensate formation while retaining gas-phase chemical equilibrium. It is a useful comparison to the retained-condensate limit; it is not a rainout calculation and does not freeze gas chemistry.</p>
{figure('condensation_control','Figure 3. Selected pressure profiles with and without equilibrium condensation. The control reveals the importance of phase availability; it does not represent a self-consistent vertical atmospheric structure.')}

<h2>Numerical evidence and practical limits</h2>
<p>The reference comparison contains 100 deep cases from the authors’ two metallicity workbooks and 50 atmospheric cases. Every reference case converges. The current code differs from the archived deep gas mole fractions by at most 2.08%, pressure by 0.154%, and SiH₄/CH₄ by 0.318%. The atmospheric ratio, when supplied with the archive’s own inventories and pressures, agrees within {reference_atm['maximum_relative_ratio_difference']:.2g}. This is a close current-version reproduction, not exact historical-code parity.</p>
<p>Accepted deep cases conserve every tracked element within {checks['meb']['maximum_relative_element_error']:.2g} relative error. The atmospheric and gas-only cases conserve them within {max(checks['atmosphere']['maximum_relative_element_error'],checks['gas_only']['maximum_relative_element_error']):.2g}. All imposed pressures and gas-reaction identities were checked. Eighteen selected deep cases were re-solved with a different numerical method, random 0.05-standard-deviation perturbations to log-mole starting values, and 100-fold tighter tolerances; maximum gas-fraction disagreement was {robustness['maximum_relative_gas_mole_fraction_difference']:.2g}.</p>
<p>Three attempted cases were excluded: 3500 K, 1 wt% H, s<sub>H₂</sub> = 0.1 at each of the three metallicities. They did not converge reliably through continuation. Their failure is not evidence of a physical spinodal. The exclusions and exact accepted arrays are recorded in <a href="outputs/study_meb.json">study_meb.json</a>.</p>
<p>Scientific limits are larger than these numerical errors. The deep model treats reactive rock as silica plus a fixed inert solvent; it is not a Mg–Fe-bearing silicate liquid solution. Empirical dissolution laws and gas equations of state are extrapolated to some study conditions. Deep pressures in the accepted grid span approximately 0.24–26 GPa. Fixed geometry and the thin-column pressure relation are inherited modeling assumptions. The archive’s core fraction is 0.295334691460966, while its separately derived rounded silicon inventory uses 0.327; both conventions are reproduced explicitly in the code.</p>
<p>The atmospheric calculations are independent closed parcels. Pure solids remain available to buffer gas chemistry. Settling and rainout, finite nucleation or reaction rates, vertical mixing, photochemistry and thermal evolution can change that availability. Consequently these curves establish sensitivity within this equilibrium network, not predicted spectra or an observable spinodal signature.</p>

<h2>What this means for the proposed spinodal extension</h2>
<p><strong>The useful target is whether demixing moves the system across a change of stable phases, and whether those phases remain chemically accessible.</strong> A large shift in interior hydrogen storage can be invisible in this ratio while changing absolute abundances. A change in Si/O/C partitioning that removes a buffering condensate can instead produce a very large ratio change.</p>
<p>The next physical calculation should supply a thermodynamically consistent H₂–silicate mixing free energy, compute coexisting phase compositions and amounts with elemental conservation, and pass the accessible gas/reservoir inventory into this tested chemistry pipeline. A constrained Gibbs-energy Hessian tests spinodal instability; common-tangent or tangent-plane equilibrium determines coexistence. Neither the hypothetical solubility multiplier nor a failed chemical solve supplies that information.</p>
<p>To isolate an effect specifically due to spinodal decomposition, compare equilibrium separation with finite-rate separation and mixing along the same cooling path. The equilibrium endpoint alone does not identify the separation mechanism. A transport/rainout control should establish whether Si and SiC remain in contact with atmospheric gas. Track individual SiH₄, CH₄, H₂O and CO abundances alongside their ratios.</p>
<p>The broad idea that silicate–hydrogen separation affects atmospheric composition already has prior work: <a href="https://arxiv.org/abs/2601.00059v2">Tejada Arevalo et al., Sub-Neptune Memories I</a> couples silicate rain to interior evolution and discusses atmospheric silicate depletion. A focused contribution here would be the reactive chemistry, phase transitions and timescale conditions that determine when an interior separation signal survives into atmospheric gas.</p>

<h2>Reproduce or extend the investigation</h2>
<p>The upstream repository remains unchanged. The extension is an isolated research harness in this directory. Raw sources retain their original SHA256 hashes. The model, study arrays, validation summaries and three figures are saved locally; plots are available as PNG, SVG and PDF.</p>
<pre>.venv/bin/python investigation/validate_reference.py
.venv/bin/python investigation/validate_atmosphere.py
.venv/bin/python investigation/run_study.py
.venv/bin/python investigation/run_atmospheres.py
.venv/bin/python investigation/check_numerics.py
.venv/bin/python investigation/analyse.py
.venv/bin/python investigation/build_report.py</pre>
<p>Run from the workspace root. See <a href="README.txt">README.txt</a> for phase names and conventions, <a href="model.py">model.py</a> for the chemical model, <a href="outputs/analysis_summary.json">analysis_summary.json</a> for selected results, <a href="sources/downloaded.json">downloaded.json</a> for archive provenance, and <a href="../environment.freeze.txt">environment.freeze.txt</a> for the installed environment.</p>
<p class="meta">Report generated {stamp}. All reported sensitivity values were computed in this workspace. The thermodynamic buffering interpretation was checked directly against the accepted outputs.</p>
</main></body></html>'''
    (HERE/'report.html').write_text(text)
    manifest=dict(generated_utc=stamp,atmodeller_commit=commit,scope='Hydrogen dissolution sensitivity with local atmospheric equilibrium; no spinodal model',
                  checks=checks,buffer_checks=buffers,reference_meb=reference,reference_atmosphere=reference_atm,
                  numerical_robustness=robustness,source_downloads=json.loads((HERE/'sources'/'downloaded.json').read_text()),files={})
    paths=[HERE/'model.py',HERE/'run_study.py',HERE/'run_atmospheres.py',HERE/'analyse.py',HERE/'check_numerics.py',HERE/'validate_reference.py',HERE/'validate_atmosphere.py',HERE/'report.html']
    paths += [OUT/name for name in ['study_meb.json','atmosphere_condensed.json','atmosphere_gas_only.json','analysis_summary.json']]
    for path in paths:manifest['files'][str(path.relative_to(HERE))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'provenance.json').write_text(json.dumps(manifest,indent=2))
    inline=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><style>body{{margin:0;font-family:var(--font-sans,system-ui);color:var(--foreground,#173244)}}p{{line-height:1.5;font-size:14px}}img{{width:100%;height:auto;display:block;background:white;border-radius:6px}}.note{{font-size:12px;color:var(--muted-foreground,#62717a)}}</style></head><body>
<p><strong>Computed with Atmodeller:</strong> the same hydrogen-storage perturbation leaves the atmospheric ratio unchanged at 1× and 10× solar, but changes its chemical regime at 100× solar.</p>
<img src="figures/phase_buffers.png" alt="Atmospheric silane to methane ratios versus hydrogen solubility multiplier, showing constant solar-composition buffers and transitions at 100 times solar.">
<p class="note">Hypothetical solubility perturbations; equilibrium condensates retained. No spinodal dynamics, rainout or detection model. All 615 targeted atmospheric scenarios converged. Lines connect sampled cases.</p></body></html>'''
    (HERE/'inline.html').write_text(inline)
    print(HERE/'report.html')


if __name__=='__main__':main()
