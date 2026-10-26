# SPDX-License-Identifier: Apache-2.0

"""Build a portable interactive view of measured study results, without a server."""

import base64
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    cases = []
    for network in ("condensed", "gas_only"):
        with (HERE / "outputs" / f"atmosphere_{network}_summary.csv").open() as stream:
            for row in csv.DictReader(stream):
                if not (
                    float(row["deep_temperature_K"]) == 3500
                    and float(row["hydrogen_wt_percent"]) in (1, 3)
                    and float(row["atmosphere_temperature_K"]) == 2500
                    and float(row["atmosphere_pressure_bar"]) == 1
                ):
                    continue
                fields = (
                    "hydrogen_wt_percent",
                    "entrained_lower_fraction",
                    "H2",
                    "H2O",
                    "SiO",
                    "SiH4",
                    "gas_mass_kg_per_kg_bulk",
                    "mean_molar_mass_g_per_mol",
                )
                cases.append(
                    {"mode": row["mode"], "network": network, **{k: float(row[k]) for k in fields}}
                )
    assert len(cases) == 20
    template = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Atmodeller equilibrium separation study</title>
<style>
body{margin:0;color:var(--foreground,#243747);font:15px/1.55 var(--font-sans,system-ui,sans-serif)}
main{width:100%;max-width:1000px;margin:auto}p{margin:0 0 14px}.lede{font-size:21px;font-weight:650;line-height:1.35}
.muted,small{color:var(--muted-foreground,#586572)}a{color:var(--foreground,#167c80);text-decoration:underline}
figure{margin:16px 0}figure img{width:100%;height:auto;display:block;border-radius:var(--radius,8px)}
figcaption{font-size:13px;margin-top:8px}.controls{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}
label{display:grid;gap:4px;font-size:13px;flex:1;min-width:150px}select{padding:8px;border-radius:6px;border:1px solid var(--border,#ced7dc);background:var(--card,#fff);color:inherit;font:inherit}
table{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}th,td{padding:9px 5px;text-align:right;border-bottom:1px solid var(--border,#d9e0e5)}th:first-child,td:first-child{text-align:left}th{font-size:12px;font-weight:600}.less{color:var(--chart-1,#167c80)}.more{color:var(--chart-2,#b36829)}
.status{padding:16px;background:var(--secondary,#edf3f5);border-radius:var(--radius,8px);margin:14px 0}.metrics{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}.metrics strong{display:block;font-size:20px}.metrics span{font-size:13px}.small{font-size:13px}@media(max-width:450px){table{font-size:12px}th,td{padding:8px 3px}.lede{font-size:19px}.metrics strong{font-size:17px}}
</style><main>
<p class="lede">Separation changes the gas supply. Entrainment and condensation determine how much of that change survives.</p>
<p class="muted">7,236 accepted equilibrium solves · H₂–MgSiO₃ binary · matched homogeneous controls</p>
<figure><img src="__FIGURE__" alt="Binary phase diagram at one gigapascal, showing the larger equilibrium coexistence region and smaller spinodal instability region. The one weight percent hydrogen line never enters the spinodal region."><figcaption>At 1 wt% H₂, separation requires nucleation in this model. At 3 wt%, the homogeneous mixture can enter the spinodal. The API's <code>spinodal</code> mode computes equilibrium throughout the coexistence gap.</figcaption></figure>
<p><strong>Explore the computed hot-parcel comparison</strong><br><span class="muted">Deep source: 3500 K, 1 GPa → atmospheric parcel: 2500 K, 1 bar.</span></p>
<div class="controls">
<label>Initial hydrogen<select id="bulk"><option value="1">1 wt% — metastable</option><option value="3">3 wt% — spinodally unstable</option></select></label>
<label>Lower phase entrained (η)<select id="eta"><option value="0">0% · upper phase only</option><option value="0.01">1%</option><option value="0.1">10%</option><option value="1">100% · full remixing</option></select></label>
<label>Atmospheric phases<select id="network"><option value="condensed">Allow condensates</option><option value="gas_only">Gas-only control</option></select></label>
</div>
<p class="small muted">Gas mole fractions; the reference uses the entire bulk inventory.</p>
<table aria-label="Gas mole fraction comparison"><thead><tr><th>Gas</th><th>Whole-bulk feed</th><th>Selected feed</th><th>Change</th></tr></thead><tbody id="rows"></tbody></table>
<div class="metrics"><div><span>Gas mass per kg of deep bulk</span><strong id="mass"></strong><small id="massRef"></small></div><div><span>Mean gas molar mass</span><strong id="molar"></strong><small id="molarRef"></small></div></div>
<p id="message" class="status" aria-live="polite"></p>
<p class="small muted">These controls re-display solved cases; no new calculation runs in the browser. Transfer is imposed and one-way. This is a hot-parcel sensitivity experiment, without a planetary cooling path, kinetic decomposition or a prediction of observable spectra. SiH₄ remains a trace constituent in the displayed case.</p>
<p class="small"><a href="https://github.com/felixlaga/atmodeller/blob/feat/spinodal-decomposition/research/spinodal_study/README.md">Full report, API example and reproduction</a> · <a href="https://arxiv.org/html/2509.13320v1#S2.SS1.SSS1">Published binary free energy</a></p>
</main><script>
const cases=__CASES__;
const species=[['H2','H₂'],['H2O','H₂O'],['SiO','SiO'],['SiH4','SiH₄']];
function fmt(n){return n<0.0001?n.toExponential(3):Number(n.toPrecision(5)).toString()}
function select(w,eta,network,mode){return cases.find(r=>r.hydrogen_wt_percent===w&&r.network===network&&r.mode===mode&&(mode==='homogeneous'||r.entrained_lower_fraction===eta))}
function render(){
 const w=Number(document.getElementById('bulk').value), eta=Number(document.getElementById('eta').value), network=document.getElementById('network').value;
 const ref=select(w,eta,network,'homogeneous'), chosen=select(w,eta,network,'spinodal');
 document.getElementById('rows').innerHTML=species.map(([key,label])=>{const q=chosen[key]/ref[key],same=Math.abs(q-1)<1e-6;const change=same?'unchanged':q<1?`${(1/q).toPrecision(3)}× lower`:`${q.toPrecision(3)}× higher`;return `<tr><td>${label}</td><td>${fmt(ref[key])}</td><td>${fmt(chosen[key])}</td><td class="${same?'':q<1?'less':'more'}">${change}</td></tr>`}).join('');
 document.getElementById('mass').textContent=fmt(chosen.gas_mass_kg_per_kg_bulk)+' kg';
 document.getElementById('massRef').textContent='Reference: '+fmt(ref.gas_mass_kg_per_kg_bulk)+' kg';
 document.getElementById('molar').textContent=fmt(chosen.mean_molar_mass_g_per_mol)+' g/mol';
 document.getElementById('molarRef').textContent='Reference: '+fmt(ref.mean_molar_mass_g_per_mol)+' g/mol';
 let text=eta===1?'Full remixing recovers the original composition and gas mass.':network==='condensed'&&w===1&&eta===0.1?'Condensation buffers the gas back to its reference composition, even though gas mass is still different.':eta===0?'Selective upper-phase transfer makes the gas more hydrogen-rich and lowers water and silicon monoxide.': 'Entraining the silicate-rich phase changes both composition and available gas mass.';
 if(network==='gas_only')text+=' Forbidding condensation can greatly inflate ratios of trace SiH₄ abundances.';
 document.getElementById('message').textContent=text;
}
for(const id of ['bulk','eta','network'])document.getElementById(id).addEventListener('change',render);
for(const w of [1,3])for(const eta of [0,.01,.1,1])for(const n of ['condensed','gas_only']){const row=select(w,eta,n,'spinodal');if(!row||!species.every(([k])=>row[k]>0&&Number.isFinite(row[k])))throw Error('Missing or invalid result');}
for(const w of [1,3])for(const eta of [0,.01,.1,1])for(const n of ['condensed','gas_only']){
 document.getElementById('bulk').value=String(w);document.getElementById('eta').value=String(eta);document.getElementById('network').value=n;document.getElementById('eta').dispatchEvent(new Event('change'));
 if(document.querySelectorAll('#rows tr').length!==4)throw Error('Missing rendered gas rows');
 if(document.getElementById('mass').textContent!==fmt(select(w,eta,n,'spinodal').gas_mass_kg_per_kg_bulk)+' kg')throw Error('Wrong rendered mass');
 if(eta===1&&[...document.querySelectorAll('#rows td:last-child')].some(e=>e.textContent!=='unchanged'))throw Error('Remixing control failed');
}
document.getElementById('bulk').value='1';document.getElementById('eta').value='0';document.getElementById('network').value='condensed';render();console.log('All 16 control-change events render archived results; full-remixing controls pass.');
</script></html>"""
    html = template.replace("__CASES__", json.dumps(cases, separators=(",", ":")))
    figure = HERE / "figures" / "phase_diagram.png"
    (HERE / "inline.html").write_text(html.replace("__FIGURE__", str(figure)))
    data_uri = "data:image/png;base64," + base64.b64encode(figure.read_bytes()).decode()
    (HERE / "summary.html").write_text(html.replace("__FIGURE__", data_uri))


if __name__ == "__main__":
    main()
