/* RecursiveNe — Nous brain visualizer.  Replay of the repo's LAST-RUN outputs (not live).
 *
 * ┌──────────────────────────────────────────────────────────────────────────────────────┐
 * │  MAINTENANCE: if the model / data systems change, edit ONLY the CONFIG + FIELDS blocks   │
 * │  below. Every visual reads its data through FIELDS (safe accessors) and its look through  │
 * │  CONFIG (tunables). Missing fields degrade gracefully and are reported in the HUD warn    │
 * │  line — nothing is silently faked. Add a new entity/field by adding an accessor here.     │
 * └──────────────────────────────────────────────────────────────────────────────────────┘
 */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

/* ============================ CONFIG — visual tunables ============================ */
const CONFIG = {
  radii:   { core: 6, shell: 9, neuron: 18, pool: 30 },
  timing:  { secPerSeason: 1.6, titleFadeMs: 3200 },
  orbit:   { rate: 0.02 },   // each concept orbits at speed = its frequency * rate (freq -> orbital freq)
  strain:  { threshold: 1000, kickMax: 0.9, chromaMax: 0.006 },   // cost_to_know above threshold = strain
  synapse: { k: 3, base: 0.05, flash: 0.7 },
  bloom:   { strength: 0.62, radius: 0.5, threshold: 0.12, strainSurge: 0.8 },
  palette: {                                                       // warmth ramp keyed to gamma
    coolA: 0x3a5bd0, coolB: 0x36c5d8, warmA: 0xffb648, warmB: 0xffd98a,
    core: 0x9a7dff, coreWarm: 0xffb648, strain: 0xff3b5c, gold: 0xffd98a, steel: 0x39414f,
    accept: 0x5fe08a, reject: 0x6a6f7a, bone: 0xc9c4b8, bg: 0x04050a,
  },
};

/* ============================ FIELDS — schema accessors =========================== */
/* Rename a field in the pipeline?  Change it here once.  Each returns a safe default.   */
const FIELDS = {
  meta:      d => d.meta || {},
  seasons:   d => d.seasons || [],
  garden:    d => d.garden || [],
  neurons:   d => d.neurons || [],
  pool:      d => d.pool || [],
  selfEdits: d => d.self_edits || [],
  race:      d => d.race || {},
  fmax:      d => d.encoder_fmax || 26,
  // per-season
  sSeason: s => s.season, sComplexity: s => s.complexity, sCost: s => s.cost_to_know,
  sTotal: s => s.total_known, sDisc: s => s.discovered_freqs, sRep: s => s.rep_size, sRidge: s => s.ridge,
  // per-neuron (NOTE the field is discovered_season, not discovered_order)
  nFreq: n => n.freq, nId: n => n.id, nDisc: n => n.discovered_season,
  // per-garden-row
  gSeason: g => g.season, gRep: g => g.repertoire, gFeat: g => g.n_features,
  gGamma: g => g.gamma, gHard: g => g.hardest_known, gAdded: g => g.added_w,
  // self-edit
  eEvent: e => e.event, eStage: e => e.stage, eTarget: e => e.target, eDescr: e => e.descr,
  eAcc: e => !!e.accepted, eBefore: e => e.meta_cost_before, eAfter: e => e.meta_cost_after,
};
const PROTECTED = ['objective.py','invariant.py','world.py','core/killswitch.py','closure/selfmod.py','closure/driver.py'];

/* HUD rows: [key, sourceLabel, description, accessor(state)] — declarative, add/remove freely. */
const HUD_ROWS = [
  ['season','seasons[i].season','replay clock', st => `${st.season} / ${st.N}`],
  ['complexity','seasons[i].complexity','frontier freq (w)', st => st.complexity],
  ['cost_to_know','seasons[i].cost_to_know','samples to master', st => st.cost],
  ['total_known','seasons[i].total_known','activities mastered', st => st.total],
  ['discovered_freqs','seasons[i].discovered_freqs','concepts in encoder', st => st.disc],
  ['rep_size','seasons[i].rep_size','working capacity (D)', st => st.rep],
  ['ridge','seasons[i].ridge','RLS prior precision', st => st.ridge],
  ['n_features','garden[k].n_features','ambition capacity (D)', st => st.gFeat],
  ['hardest_known','garden[k].hardest_known','open-ended frontier', st => st.gHard],
  ['gamma','garden[k].gamma','RFF bandwidth', st => st.gGamma != null ? (+st.gGamma).toFixed(3) : '–'],
  ['repertoire','garden[k].repertoire','garden repertoire', st => st.gRep],
];

/* ============================ load + validate ==================================== */
const D = window.BRAIN_DATA;
const errEl = document.getElementById('err');
if (!D) { errEl.style.display = 'flex'; throw new Error('no BRAIN_DATA'); }

const seasons = FIELDS.seasons(D), garden = FIELDS.garden(D), neurons = FIELDS.neurons(D), pool = FIELDS.pool(D);
const N = seasons.length, G = garden.length, NN = neurons.length;
const FMAX = FIELDS.fmax(D), FMIN = 1.0;
const warns = [];
if (!N) warns.push('no seasons[]'); if (!NN) warns.push('no neurons[]'); if (!G) warns.push('no garden[]');
if (FIELDS.meta(D).live === true) warns.push('meta.live=true (expected replay)');
if (warns.length) { const w = document.getElementById('warn'); w.style.display = 'block';
  w.textContent = 'schema note: ' + warns.join(' · '); }

const norm = f => (f - FMIN) / (FMAX - FMIN);
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/* ============================ scene ============================================== */
const app = document.getElementById('app');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(CONFIG.palette.bg, 0.010);
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 3000);
camera.position.set(0, 7, 66);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
app.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.dampingFactor = .07; controls.minDistance = 24; controls.maxDistance = 150;
controls.autoRotate = true; controls.autoRotateSpeed = 0.35;
renderer.domElement.addEventListener('pointerdown', () => controls.autoRotate = false);

const root = new THREE.Group(); scene.add(root);
const { radii: R, palette: P } = CONFIG;

/* nebula backdrop */
{ const m = new THREE.ShaderMaterial({ side: THREE.BackSide, uniforms: {},
    vertexShader: `varying vec3 p; void main(){ p=position; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
    fragmentShader: `varying vec3 p; void main(){ float t=clamp(p.y/1000.*.5+.5,0.,1.);
      float r=length(p.xy)/1400.; vec3 c=mix(vec3(.015,.02,.038), vec3(.05,.06,.11), t);
      c+=vec3(.03,.02,.05)*smoothstep(1.,.2,r); gl_FragColor=vec4(c,1.); }` });
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(1200, 24, 16), m)); }

/* ambient dust (framing, not data — labeled in legend) */
{ const n = 1400, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) { const r = 40 + Math.random() * 900, u = Math.random() * 2 - 1, th = Math.random() * 6.28;
    const s = Math.sqrt(1 - u * u); pos[i*3]=Math.cos(th)*s*r; pos[i*3+1]=u*r; pos[i*3+2]=Math.sin(th)*s*r; }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0x4a5680, size: 0.7, transparent: true, opacity: .5, sizeAttenuation: true }))); }

/* ---- neurons : one Points, freq-sorted Fibonacci sphere ---- */
const order = neurons.map((_, i) => i).sort((a, b) => FIELDS.nFreq(neurons[a]) - FIELDS.nFreq(neurons[b]));
const ramp = t => { const c1=new THREE.Color(P.coolA),c2=new THREE.Color(P.coolB),c3=new THREE.Color(P.warmA),c4=new THREE.Color(P.warmB);
  return t<.5 ? c1.lerp(c2,t/.5) : c3.lerp(c4,(t-.5)/.5); };
const npos = new Float32Array(NN*3), ncol = new Float32Array(NN*3), nsize = new Float32Array(NN), nalpha = new Float32Array(NN), nseed = new Float32Array(NN);
const meta = [];
for (let k = 0; k < NN; k++) {
  const nu = neurons[order[k]]; const t = norm(FIELDS.nFreq(nu));
  const y = NN>1 ? (k/(NN-1))*2-1 : 0; const r = Math.sqrt(Math.max(0,1-y*y)); const phi = k*2.39996323;
  const rad = R.neuron*(0.82+0.30*t);
  const v = new THREE.Vector3(Math.cos(phi)*r*rad, y*rad, Math.sin(phi)*r*rad);
  npos[k*3]=v.x; npos[k*3+1]=v.y; npos[k*3+2]=v.z;
  const c = ramp(t); ncol[k*3]=c.r; ncol[k*3+1]=c.g; ncol[k*3+2]=c.b;
  nsize[k]=3.0; nalpha[k]=0; nseed[k]=Math.random()*6.28;
  meta.push({ id: FIELDS.nId(nu), freq: FIELDS.nFreq(nu), disc: FIELDS.nDisc(nu), pos: v, known: false,
    rxz: Math.hypot(v.x, v.z), y0: v.y, phi0: Math.atan2(v.z, v.x) });
}
const ngeo = new THREE.BufferGeometry();
ngeo.setAttribute('position', new THREE.BufferAttribute(npos,3));
ngeo.setAttribute('aColor', new THREE.BufferAttribute(ncol,3));
ngeo.setAttribute('aSize', new THREE.BufferAttribute(nsize,1));
ngeo.setAttribute('aAlpha', new THREE.BufferAttribute(nalpha,1));
ngeo.setAttribute('aSeed', new THREE.BufferAttribute(nseed,1));
const nmat = new THREE.ShaderMaterial({ transparent:true, depthWrite:false, blending:THREE.AdditiveBlending,
  uniforms:{ uScale:{value:innerHeight/2}, uTime:{value:0} },
  vertexShader:`attribute vec3 aColor; attribute float aSize; attribute float aAlpha; attribute float aSeed;
    varying vec3 vC; varying float vA; uniform float uScale; uniform float uTime;
    void main(){ vC=aColor; float tw=0.85+0.15*sin(uTime*2.0+aSeed); vA=aAlpha*tw;
      vec4 mv=modelViewMatrix*vec4(position,1.); gl_PointSize=aSize*tw*uScale/(-mv.z); gl_Position=projectionMatrix*mv; }`,
  fragmentShader:`varying vec3 vC; varying float vA;
    void main(){ float d=length(gl_PointCoord-vec2(.5)); float core=smoothstep(.5,.0,d);
      float halo=smoothstep(.5,.18,d); gl_FragColor=vec4(vC*(0.5+core), (halo*0.5+core)*vA); }` });
const neuronPoints = new THREE.Points(ngeo, nmat); root.add(neuronPoints);

/* ---- synapses : kNN edges, per-vertex alpha flash ---- */
const edges = [];
for (let k = 0; k < NN; k++) {
  const d = meta.map((m, j) => [j, meta[k].pos.distanceTo(m.pos)]).filter(x => x[0] !== k).sort((a,b)=>a[1]-b[1]);
  for (let e = 0; e < Math.min(CONFIG.synapse.k, d.length); e++) { const j = d[e][0];
    if (!edges.some(x => (x.a===j&&x.b===k))) edges.push({ a:k, b:j }); }
}
const E = edges.length;
const epos = new Float32Array(E*6), ealpha = new Float32Array(E*2);
for (let i=0;i<E;i++){ const a=meta[edges[i].a].pos, b=meta[edges[i].b].pos;
  epos.set([a.x,a.y,a.z,b.x,b.y,b.z], i*6); }
const egeo = new THREE.BufferGeometry();
egeo.setAttribute('position', new THREE.BufferAttribute(epos,3));
egeo.setAttribute('aAlpha', new THREE.BufferAttribute(ealpha,1));
const emat = new THREE.ShaderMaterial({ transparent:true, depthWrite:false, blending:THREE.AdditiveBlending,
  uniforms:{ uColor:{value:new THREE.Color(0x6f86ff)} },
  vertexShader:`attribute float aAlpha; varying float vA; void main(){ vA=aAlpha; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader:`uniform vec3 uColor; varying float vA; void main(){ gl_FragColor=vec4(uColor, vA); }` });
const synapses = new THREE.LineSegments(egeo, emat); root.add(synapses);

/* ---- traveling synapse pulses ---- */
const MAXP = 90; const ppos = new Float32Array(MAXP*3), pa = new Float32Array(MAXP), pc = new Float32Array(MAXP*3);
const pulses = []; // {a,b,t,sp,col}
const pgeo = new THREE.BufferGeometry();
pgeo.setAttribute('position', new THREE.BufferAttribute(ppos,3));
pgeo.setAttribute('aColor', new THREE.BufferAttribute(pc,3));
pgeo.setAttribute('aSize', new THREE.BufferAttribute(new Float32Array(MAXP).fill(5),1));
pgeo.setAttribute('aAlpha', new THREE.BufferAttribute(pa,1));
pgeo.setAttribute('aSeed', new THREE.BufferAttribute(new Float32Array(MAXP),1));
const pulsePts = new THREE.Points(pgeo, nmat.clone()); pulsePts.material.uniforms.uScale = nmat.uniforms.uScale;
pulsePts.material.uniforms.uTime = nmat.uniforms.uTime; root.add(pulsePts);

/* ---- core + inner glow + feature shell ---- */
const core = new THREE.Mesh(new THREE.IcosahedronGeometry(1,1),
  new THREE.MeshBasicMaterial({ color:P.core, wireframe:true, transparent:true, opacity:.55 }));
root.add(core);
const glow = new THREE.Mesh(new THREE.IcosahedronGeometry(1,2), new THREE.ShaderMaterial({
  transparent:true, blending:THREE.AdditiveBlending, side:THREE.FrontSide, depthWrite:false,
  uniforms:{ uColor:{value:new THREE.Color(P.core)}, uPow:{value:2.2} },
  vertexShader:`varying vec3 vN; varying vec3 vV; void main(){ vN=normalize(normalMatrix*normal);
    vec4 mv=modelViewMatrix*vec4(position,1.); vV=normalize(-mv.xyz); gl_Position=projectionMatrix*mv; }`,
  fragmentShader:`uniform vec3 uColor; uniform float uPow; varying vec3 vN; varying vec3 vV;
    void main(){ float f=pow(1.-max(dot(vN,vV),0.), uPow); gl_FragColor=vec4(uColor, f*0.7); }` }));
root.add(glow);
const shell = new THREE.Mesh(new THREE.SphereGeometry(1,40,28),
  new THREE.MeshBasicMaterial({ color:0x6a7bff, transparent:true, opacity:.05, side:THREE.BackSide })); root.add(shell);

/* ---- garden growth rings (repertoire) ---- */
const ringsG = new THREE.Group(); root.add(ringsG);
const REP_MAX = Math.max(1, ...garden.map(g => FIELDS.gRep(g) || 0));
for (let i = 0; i < REP_MAX; i++) { const rr = R.core+1 + i*0.4;
  const ring = new THREE.Mesh(new THREE.TorusGeometry(rr, 0.025, 6, 120),
    new THREE.MeshBasicMaterial({ color:P.warmA, transparent:true, opacity:0 }));
  ring.rotation.x = Math.PI/2; ringsG.add(ring); }

/* ---- pool target rings (goals) ---- */
const poolRings = pool.map(f => { const y = (norm(f)*2-1)*R.pool*0.62;
  const rr = Math.sqrt(Math.max(0.04, 1-(y/(R.pool*0.8))**2))*R.pool*0.46+2;
  const m = new THREE.Mesh(new THREE.TorusGeometry(rr, 0.05, 8, 90),
    new THREE.MeshBasicMaterial({ color:P.steel, transparent:true, opacity:.4 }));
  m.position.y = y; m.rotation.x = Math.PI/2; root.add(m); return { mesh:m, freq:f, met:false }; });

/* ---- RULER (protected, matte, never glows, non-interactive) ---- */
{ const cv=document.createElement('canvas'); cv.width=1024; cv.height=64; const x=cv.getContext('2d');
  x.fillStyle='#090a10'; x.fillRect(0,0,1024,64); x.strokeStyle='#c9c4b8'; x.lineWidth=2; x.strokeRect(2,2,1020,60);
  x.fillStyle='#c9c4b8'; x.font='24px monospace'; x.textBaseline='middle';
  x.fillText('RULER  ·  TAU = 0.05  ·  protected  ·  the system cannot edit success', 24, 34);
  const band=new THREE.Mesh(new THREE.PlaneGeometry(56,3.5), new THREE.MeshBasicMaterial({ map:new THREE.CanvasTexture(cv), transparent:true }));
  band.position.set(0,-R.pool*0.9,0); scene.add(band); }

/* ============================ post-processing =================================== */
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), CONFIG.bloom.strength, CONFIG.bloom.radius, CONFIG.bloom.threshold);
composer.addPass(bloom);
const GradeShader = {
  uniforms:{ tDiffuse:{value:null}, uStrain:{value:0}, uTime:{value:0}, uChroma:{value:CONFIG.strain.chromaMax} },
  vertexShader:`varying vec2 vUv; void main(){ vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.); }`,
  fragmentShader:`uniform sampler2D tDiffuse; uniform float uStrain; uniform float uTime; uniform float uChroma;
    varying vec2 vUv;
    void main(){ vec2 d=vUv-0.5; float off=uChroma*uStrain*(0.4+length(d));
      vec3 c; c.r=texture2D(tDiffuse, vUv+d*off).r; c.g=texture2D(tDiffuse,vUv).g; c.b=texture2D(tDiffuse, vUv-d*off).b;
      float vig=smoothstep(1.15,0.35,length(d)); c*=mix(vig, vig*1.0, 1.0);
      c+=vec3(0.10,0.0,0.02)*uStrain*smoothstep(0.2,0.9,length(d));   // crimson edge wash on strain
      float g=fract(sin(dot(vUv*uTime, vec2(12.9898,78.233)))*43758.5453); c+=(g-0.5)*0.025;
      gl_FragColor=vec4(c,1.); } ` };
const grade = new ShaderPass(GradeShader); grade.renderToScreen = true; composer.addPass(grade);
let useBloom = innerWidth > 760;

/* ============================ HUD / feed / race / legend ======================== */
const rowsEl = document.getElementById('rows'); const rowEl = {};
HUD_ROWS.forEach(([key, src, desc, fn]) => { const d = document.createElement('div'); d.className='row';
  d.innerHTML = `<span class="k">${key}</span><span class="v" id="v-${key}">–</span>`;
  d.addEventListener('mousemove', e => tipAt(e, `${src}<br><span style="color:#838fab">${desc}</span><br>= <b>${d._src||''}</b>`));
  d.addEventListener('mouseleave', tipHide); rowsEl.appendChild(d); rowEl[key] = { el:d, fn, src }; });
document.getElementById('telos').textContent = FIELDS.meta(D).telos || '';

let editEls = [];
(function feed(){ const el = document.getElementById('edits');
  FIELDS.selfEdits(D).forEach(e => { const acc = FIELDS.eAcc(e); const tgt = FIELDS.eTarget(e)||'';
    const tamper = !acc && PROTECTED.some(p => tgt.includes(p));
    const div = document.createElement('div'); div.className = 'edit ' + (acc?'acc':(tamper?'tamper':'rej'));
    let delta=''; const b=FIELDS.eBefore(e), a=FIELDS.eAfter(e);
    if (acc && b && a) delta = ` <span class="delta">${(b/1e6).toFixed(1)}M→${(a/1e6).toFixed(1)}M</span>`;
    div.innerHTML = `<span class="t">${FIELDS.eStage(e)||'?'} · ${tgt}</span><br><span class="d">${FIELDS.eDescr(e)||FIELDS.eEvent(e)}</span>${delta}`
      + (tamper ? `<br><span class="tag">⛔ GATE REJECTED: protected file</span>` : '');
    el.appendChild(div); editEls.push(div); }); })();

(function race(){ const el = document.getElementById('racebody'); const r = FIELDS.race(D), r6 = r['6knob'], r3 = r['3knob'];
  const fmt = v => (v==null || !isFinite(v)) ? '∞ never reached' : (v/1e6).toFixed(2)+'M';
  const bars = (rr, label) => { if (!rr) return '';
    const c = rr.claude_best, s = rr.rsi_best, lo = Math.min(isFinite(c)?c:Infinity, isFinite(s)?s:Infinity);
    const wc = isFinite(c) ? Math.max(8, 100*lo/c) : 8, ws = isFinite(s) ? Math.max(8, 100*lo/s) : 8;
    return `<div style="font-size:10.5px;color:#fff;margin:2px 0">${label}: <b>${rr.verdict}</b></div>
      <div class="bar"><span style="width:${wc}%;background:#36c5d8"></span><span class="lab">Claude ${fmt(c)}</span></div>
      <div class="bar"><span style="width:${ws}%;background:#ffd98a"></span><span class="lab">RSI ${fmt(s)}</span></div>`; };
  el.innerHTML = bars(r6,'6-knob self-edit space') + bars(r3,'3-knob self-edit space')
    + `<div style="font-size:10px;color:#838fab">lower = cheaper held-out cost. learned self-edit search wins as the space grows (6-knob); human hand-tuning wins in low-dim (3-knob).</div>`; })();

document.getElementById('legend').innerHTML = `<b>legend — every element ↔ a logged field</b><br>
  <span class="sw" style="background:#36c5d8"></span>neuron = discovered encoder freq (concept)<br>
  <span class="sw" style="background:#6f86ff"></span>synapse = nearest-concept link; fires on a 'knew' event<br>
  <span class="sw" style="background:#ffd98a"></span>pool ring = goal freq; gold = reached<br>
  <span class="sw" style="background:#9a7dff"></span>core = rep_size · shell = garden.n_features<br>
  <span class="sw" style="background:#ff3b5c"></span>crimson flare = cost_to_know strain spike<br>
  <span class="sw" style="background:#c9c4b8"></span>RULER = protected TAU (never glows)<br>
  <span style="color:#9aa6c5">· orrery: each concept orbits at speed = its own frequency</span><br>
  <span style="color:#4a5680">· faint dust = ambient framing (not data)</span>`;

/* ============================ tooltip + raycast ================================= */
const tip = document.getElementById('tip');
function tipAt(e, html){ tip.innerHTML=html; tip.style.display='block';
  tip.style.left = Math.min(e.clientX+14, innerWidth-270)+'px'; tip.style.top = (e.clientY+14)+'px'; }
function tipHide(){ tip.style.display='none'; }
const ray = new THREE.Raycaster(); ray.params.Points.threshold = 1.3; const mouse = new THREE.Vector2();
renderer.domElement.addEventListener('mousemove', e => { mouse.x=(e.clientX/innerWidth)*2-1; mouse.y=-(e.clientY/innerHeight)*2+1;
  ray.setFromCamera(mouse, camera); const h = ray.intersectObject(neuronPoints);
  if (h.length){ const m = meta[h[0].index]; tipAt(e, `neuron #${m.id} · f=${m.freq.toFixed(2)}<br><span style="color:#838fab">discovered season ${m.disc}${m.known?' · known':''}</span>`); return; }
  const hp = ray.intersectObjects(poolRings.map(p=>p.mesh));
  if (hp.length){ const pr = poolRings.find(p=>p.mesh===hp[0].object); tipAt(e, `goal w=${pr.freq}<br><span style="color:#838fab">${pr.met?'reached':'not yet reached'}</span>`); return; }
  if (tip.innerHTML.startsWith('neuron')||tip.innerHTML.startsWith('goal')) tipHide(); });

/* ============================ state engine (continuous) ========================= */
const seasonVal = (i, f) => f(seasons[clamp(i,0,N-1)]);
const gardenIdx = clock => clamp(Math.round((clock-1)/Math.max(1,N-1)*(G-1)), 0, G-1);
function nearest(freq){ let b=0,bd=1e9; for(let k=0;k<NN;k++){ const d=Math.abs(meta[k].freq-freq); if(d<bd){bd=d;b=k;} } return b; }

let strainEnv = 0;            // 0..1 envelope for chromatic/bloom/shake on strain
let kick = 0;

function applyState(clock){
  const i0 = clamp(Math.floor(clock)-1, 0, N-1), i1 = clamp(i0+1, 0, N-1), fr = clamp(clock-Math.floor(clock),0,1);
  const s = seasons[i0];
  const complexity = lerp(seasonVal(i0,FIELDS.sComplexity), seasonVal(i1,FIELDS.sComplexity), fr);
  const rep = lerp(seasonVal(i0,FIELDS.sRep), seasonVal(i1,FIELDS.sRep), fr);
  const gi0 = gardenIdx(clock), gi1 = clamp(gi0+1,0,G-1);
  const gFeat = lerp(FIELDS.gFeat(garden[gi0]), FIELDS.gFeat(garden[gi1]), fr);
  const gGamma = lerp(FIELDS.gGamma(garden[gi0]), FIELDS.gGamma(garden[gi1]), fr);
  const gRep = FIELDS.gRep(garden[gi0]);
  // HUD (discrete values from the current integer season for honesty)
  const st = { N, season:FIELDS.sSeason(s), complexity:FIELDS.sComplexity(s), cost:FIELDS.sCost(s),
    total:FIELDS.sTotal(s), disc:FIELDS.sDisc(s), rep:FIELDS.sRep(s), ridge:FIELDS.sRidge(s),
    gFeat:FIELDS.gFeat(garden[gi0]), gHard:FIELDS.gHard(garden[gi0]), gGamma:FIELDS.gGamma(garden[gi0]), gRep };
  HUD_ROWS.forEach(([key,src,desc,fn]) => { const r = rowEl[key]; const v = fn(st);
    document.getElementById('v-'+key).textContent = v;
    r._src = (key in {n_features:1,hardest_known:1,gamma:1,repertoire:1})
      ? `garden[${gi0}].${key.replace('gamma','gamma')} = ${v} (garden season ${FIELDS.gSeason(garden[gi0])})`
      : `seasons[${i0}].${key} = ${v}`;
    r.el.classList.toggle('strain', key==='cost_to_know' && FIELDS.sCost(s)>CONFIG.strain.threshold); });
  const strain = FIELDS.sCost(s) > CONFIG.strain.threshold;
  document.getElementById('seasonlab').innerHTML = `season <b>${FIELDS.sSeason(s)}</b>/${N}` + (strain?` · <span style="color:var(--crimson)">STRAIN</span>`:'');
  document.getElementById('scrub').value = FIELDS.sSeason(s);
  // continuous visuals
  const focus = nearest(complexity);
  for (let k=0;k<NN;k++){ const m=meta[k]; const vis = (FIELDS.sSeason(s)>=m.disc);
    nalpha[k] = vis ? (m.known?1.0:0.45) : 0.0; nsize[k] = m.known?4.2:3.0; }
  ngeo.attributes.aAlpha.needsUpdate = ngeo.attributes.aSize.needsUpdate = true;
  // synapse base visibility (both endpoints discovered)
  for (let i=0;i<E;i++){ const va = meta[edges[i].a].disc<=FIELDS.sSeason(s), vb = meta[edges[i].b].disc<=FIELDS.sSeason(s);
    const base = (va&&vb)?CONFIG.synapse.base:0; if (ealpha[i*2] < base) { ealpha[i*2]=base; ealpha[i*2+1]=base; } }
  // core + shell
  core.scale.setScalar(0.6 + (rep/96)*R.core*0.16); glow.scale.copy(core.scale).multiplyScalar(1.18);
  shell.scale.setScalar(R.shell*(0.7 + (gFeat/96)*0.5));
  const warm = clamp((gGamma-8)/(22.85-8),0,1);
  const cc = new THREE.Color(P.core).lerp(new THREE.Color(P.coreWarm), warm*0.55);
  core.material.color.copy(cc); glow.material.uniforms.uColor.value.copy(cc);
  // growth rings
  ringsG.children.forEach((ring,idx)=> ring.material.opacity = idx<gRep ? 0.25+0.35*(idx/Math.max(1,gRep)) : 0);
  // pool rings
  poolRings.forEach(pr=>{ const met = FIELDS.sComplexity(s) >= pr.freq-0.001; pr.met=met;
    pr.mesh.material.color.setHex(met?P.gold:P.steel); pr.mesh.material.opacity = met?0.9:0.4; });
  return { focus, strain };
}

let lastInt = -1;
function fireEvents(clock){ const i = clamp(Math.round(clock)-1,0,N-1); if (i===lastInt) return; lastInt = i;
  const s = seasons[i]; const k = nearest(FIELDS.sComplexity(s));
  if (FIELDS.sSeason(s) >= meta[k].disc) meta[k].known = true;
  const cost = FIELDS.sCost(s), strain = cost>CONFIG.strain.threshold;
  spawnPulse(meta[k].pos, strain); flashSynapses(k);
  for (let e=0;e<E;e++){ if (edges[e].a===k||edges[e].b===k){ const o = edges[e].a===k?edges[e].b:edges[e].a;
    if (meta[o].disc<=FIELDS.sSeason(s)) addTravel(meta[k].pos, meta[o].pos, strain); } }
  if (strain){ strainEnv = 1.0; kick = CONFIG.strain.kickMax; }
  // feed highlight
  editEls.forEach(d=>d.classList.remove('live'));
}
function flashSynapses(k){ for (let e=0;e<E;e++){ if (edges[e].a===k||edges[e].b===k){ ealpha[e*2]=CONFIG.synapse.flash; ealpha[e*2+1]=CONFIG.synapse.flash; } } }
function spawnPulse(at, strain){ fx.push({ at:at.clone(), t:0, life:strain?1.7:0.9, strain }); }
function addTravel(a,b,strain){ if (pulses.length>=MAXP) return; pulses.push({ a:a.clone(), b:b.clone(), t:0, sp:1.6+Math.random()*0.8, strain }); }

/* expanding-ring + shockwave fx */
const fx = []; const fxRings = [];
function stepFx(dt){
  // event flares (expanding rings + strain shockwave)
  for (let i=fx.length-1;i>=0;i--){ const f=fx[i]; if (!f.m){ const col=f.strain?P.strain:P.gold;
      f.m = new THREE.Mesh(new THREE.RingGeometry(0.2,0.5,48), new THREE.MeshBasicMaterial({ color:col, transparent:true, opacity:.9, side:THREE.DoubleSide, blending:THREE.AdditiveBlending, depthWrite:false }));
      f.m.position.copy(f.at); scene.add(f.m);
      if (f.strain){ f.sw = new THREE.Mesh(new THREE.SphereGeometry(1,28,18), new THREE.MeshBasicMaterial({ color:P.strain, transparent:true, opacity:.5, wireframe:true })); scene.add(f.sw); } }
    f.t+=dt; const u=f.t/f.life; const e=1-Math.pow(1-u,3);
    f.m.lookAt(camera.position); f.m.scale.setScalar(0.6+e*(f.strain?7:3)); f.m.material.opacity=.9*(1-u);
    if (f.sw){ f.sw.scale.setScalar(2+e*22); f.sw.material.opacity=.5*(1-u); }
    if (u>=1){ scene.remove(f.m); f.m.geometry.dispose(); f.m.material.dispose(); if(f.sw){scene.remove(f.sw);f.sw.geometry.dispose();f.sw.material.dispose();} fx.splice(i,1); } }
  // traveling pulses
  let n=0; for (let i=pulses.length-1;i>=0;i--){ const p=pulses[i]; p.t+=dt*p.sp; if (p.t>=1){ pulses.splice(i,1); continue; }
    if (n<MAXP){ const x=lerp(p.a.x,p.b.x,p.t), y=lerp(p.a.y,p.b.y,p.t), z=lerp(p.a.z,p.b.z,p.t);
      ppos[n*3]=x; ppos[n*3+1]=y; ppos[n*3+2]=z; pa[n]=1-Math.abs(p.t-0.5)*2; const c=new THREE.Color(p.strain?P.strain:0xfff1c0);
      pc[n*3]=c.r; pc[n*3+1]=c.g; pc[n*3+2]=c.b; n++; } }
  for (let i=n;i<MAXP;i++) pa[i]=0;
  pgeo.attributes.position.needsUpdate = pgeo.attributes.aAlpha.needsUpdate = pgeo.attributes.aColor.needsUpdate = true;
  // decay synapse flashes toward base
  for (let i=0;i<E;i++){ const base = ealpha[i*2]>0?CONFIG.synapse.base:0; ealpha[i*2]=Math.max(0, ealpha[i*2]-dt*0.9); ealpha[i*2+1]=ealpha[i*2]; }
  egeo.attributes.aAlpha.needsUpdate = true;
}

/* ============================ transport ======================================== */
let playing = true, speed = 1, clock = 1;
const scrub = document.getElementById('scrub'); scrub.max=N; scrub.min=1; scrub.value=1;
const playBtn = document.getElementById('play'), speedBtn = document.getElementById('speed');
playBtn.onclick = () => { playing=!playing; playBtn.textContent = playing?'❚❚ pause':'▶ play'; };
speedBtn.onclick = () => { speed = speed===1?2: speed===2?4: speed===4?0.5:1; speedBtn.textContent = speed+'×'; };
document.getElementById('step+').onclick = () => { playing=false; playBtn.textContent='▶ play'; clock=Math.min(N,Math.round(clock)+1); fireEvents(clock); };
document.getElementById('step-').onclick = () => { playing=false; playBtn.textContent='▶ play'; clock=Math.max(1,Math.round(clock)-1); lastInt=-1; };
document.getElementById('reset').onclick = () => { camera.position.set(0,7,66); controls.target.set(0,0,0); controls.autoRotate=true; };
document.getElementById('spike').onclick = () => { playing=false; playBtn.textContent='▶ play';
  const k = seasons.findIndex(s=>FIELDS.sCost(s)>CONFIG.strain.threshold); clock = k>=0?FIELDS.sSeason(seasons[k]):14; lastInt=-1; fireEvents(clock); };
scrub.oninput = () => { playing=false; playBtn.textContent='▶ play'; clock=parseInt(scrub.value); lastInt=-1; };
const chip = (id, fn) => { const c=document.getElementById(id); c.onclick=()=>{ c.classList.toggle('on'); fn(c.classList.contains('on')); }; };
chip('c-race', on => document.getElementById('race').style.display = on?'block':'none');
chip('c-edits', on => document.getElementById('feed').style.display = on?'block':'none');
chip('c-labels', on => document.getElementById('legend').style.display = on?'block':'none');
document.getElementById('c-lite').onclick = function(){ this.classList.toggle('on'); useBloom = !this.classList.contains('on'); };
addEventListener('keydown', e => { if (e.code==='Space'){ e.preventDefault(); playBtn.click(); }
  else if (e.code==='ArrowRight') document.getElementById('step+').click();
  else if (e.code==='ArrowLeft') document.getElementById('step-').click();
  else if (e.key==='r'||e.key==='R') document.getElementById('reset').click();
  else if (e.key==='s'||e.key==='S') document.getElementById('spike').click(); });

/* title card fade */
setTimeout(()=>{ const t=document.getElementById('title-card'); if(t) t.style.opacity='0'; }, CONFIG.timing.titleFadeMs);

/* ============================ loop ============================================= */
const sparkDraw = () => { const cv=document.getElementById('spark'); const w=cv.clientWidth||860,h=42;
  cv.width=w*devicePixelRatio; cv.height=h*devicePixelRatio; const x=cv.getContext('2d'); x.scale(devicePixelRatio,devicePixelRatio); x.clearRect(0,0,w,h);
  const mx=Math.log10(Math.max(...seasons.map(s=>FIELDS.sCost(s)))); x.beginPath();
  seasons.forEach((s,idx)=>{ const px=idx/(N-1)*w, v=Math.log10(Math.max(10,FIELDS.sCost(s))), py=h-(v/mx)*(h-7)-3; idx?x.lineTo(px,py):x.moveTo(px,py); });
  x.strokeStyle='rgba(150,170,220,.45)'; x.lineWidth=1.2; x.stroke();
  seasons.forEach((s,idx)=>{ if (FIELDS.sCost(s)>CONFIG.strain.threshold){ const px=idx/(N-1)*w, v=Math.log10(FIELDS.sCost(s)), py=h-(v/mx)*(h-7)-3;
    x.fillStyle='#ff3b5c'; x.beginPath(); x.arc(px,py,2.4,0,7); x.fill(); } }); };
sparkDraw();

let prev = performance.now();
applyState(1); fireEvents(1);
function loop(now){ const dt=Math.min(0.05,(now-prev)/1000); prev=now; const ts=now*0.001; nmat.uniforms.uTime.value = ts;
  if (playing){ clock += dt*speed/CONFIG.timing.secPerSeason; if (clock>=N+0.999){ clock=1; lastInt=-1; meta.forEach(m=>m.known=false); }
    fireEvents(clock); }
  applyState(clock);
  // orrery: each concept orbits at speed = its own frequency (freq -> orbital frequency)
  for (let k=0;k<NN;k++){ const m=meta[k]; const ang=m.phi0 + m.freq*CONFIG.orbit.rate*ts;
    m.pos.set(Math.cos(ang)*m.rxz, m.y0, Math.sin(ang)*m.rxz);
    npos[k*3]=m.pos.x; npos[k*3+1]=m.pos.y; npos[k*3+2]=m.pos.z; }
  ngeo.attributes.position.needsUpdate = true;
  for (let i=0;i<E;i++){ const a=meta[edges[i].a].pos, b=meta[edges[i].b].pos;
    epos[i*6]=a.x;epos[i*6+1]=a.y;epos[i*6+2]=a.z; epos[i*6+3]=b.x;epos[i*6+4]=b.y;epos[i*6+5]=b.z; }
  egeo.attributes.position.needsUpdate = true;
  stepFx(dt); controls.update();
  glow.rotation.y += dt*0.15; core.rotation.y -= dt*0.2; core.rotation.x += dt*0.05;
  // strain envelope -> bloom surge + chromatic + camera kick
  strainEnv = Math.max(0, strainEnv - dt*0.7); kick = Math.max(0, kick - dt*1.7);
  bloom.strength = CONFIG.bloom.strength + strainEnv*CONFIG.bloom.strainSurge;
  grade.uniforms.uStrain.value = strainEnv; grade.uniforms.uTime.value = now*0.001;
  if (kick>0){ const a=kick*0.5; camera.position.x+=(Math.random()-.5)*a; camera.position.y+=(Math.random()-.5)*a; }
  if (useBloom) composer.render(); else renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

addEventListener('resize', () => { camera.aspect=innerWidth/innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight); composer.setSize(innerWidth,innerHeight); bloom.setSize(innerWidth,innerHeight);
  nmat.uniforms.uScale.value=innerHeight/2; sparkDraw(); });
