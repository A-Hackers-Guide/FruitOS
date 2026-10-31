import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const $ = (s) => document.querySelector(s);
const api = (p, body) => fetch(p, body ? { method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body) } : undefined).then((r) => (p === "/api/points" ? r.arrayBuffer() : r.json()));

// ---------------------------------------------------------------- colours by brain part
const GROUPS = [
  { name: "Optic lobes (vision)", test: (s) => s.startsWith("ol_") || s.startsWith("visual"), color: "#4c8dff" },
  { name: "Central brain", test: (s) => s.startsWith("cb_intrinsic"), color: "#b07cff" },
  { name: "Nerve cord (body control)", test: (s) => s.startsWith("vnc_intrinsic"), color: "#3ecf8e" },
  { name: "Sensory neurons", test: (s) => s.includes("sensory"), color: "#ffd166" },
  { name: "Descending (commands)", test: (s) => s.startsWith("descending"), color: "#ff6b4a" },
  { name: "Ascending (body → brain)", test: (s) => s.startsWith("ascending"), color: "#2ec4d6" },
  { name: "Motor & other", test: () => true, color: "#9aa3b5" },
];

let info, N, pos, groupOf, geom, glow, dim, scene, camera, renderer, controls, points;
const hidden = new Set();

async function boot() {
  info = await api("/api/info");
  N = info.neurons;
  $("#counts").textContent = `${N.toLocaleString()} neurons · ${info.connections.toLocaleString()} connections · ${info.synapses.toLocaleString()} synapses`;
  $("#credit").textContent = info.data_credit;
  const buf = await api("/api/points");
  pos = new Float32Array(buf, 0, N * 3);
  const sc = new Uint8Array(buf, N * 12, N);
  groupOf = new Uint8Array(N);
  for (let i = 0; i < N; i++) groupOf[i] = GROUPS.findIndex((g) => g.test(info.superclass[sc[i]]));
  initScene();
  initLegend();
  initSenses();
  initTabs();
  initChat();
  initLessons();
  initExplore();
  live();
  animate();
}

// ---------------------------------------------------------------- 3D brain
function initScene() {
  const canvas = $("#brain");
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  scene = new THREE.Scene();
  scene.background = new THREE.Color("#0b0d12");
  camera = new THREE.PerspectiveCamera(45, 1, 0.01, 50);
  camera.position.set(0, -0.15, 2.0);
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  geom = new THREE.BufferGeometry();
  const p = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) { p[3 * i] = pos[3 * i]; p[3 * i + 1] = -pos[3 * i + 1]; p[3 * i + 2] = -pos[3 * i + 2]; }
  geom.setAttribute("position", new THREE.BufferAttribute(p, 3));
  const col = new Float32Array(N * 3);
  const c = new THREE.Color();
  for (let i = 0; i < N; i++) { c.set(GROUPS[groupOf[i]].color); col.set([c.r, c.g, c.b], 3 * i); }
  geom.setAttribute("color", new THREE.BufferAttribute(col, 3));
  glow = new Float32Array(N); dim = new Float32Array(N);
  geom.setAttribute("glow", new THREE.BufferAttribute(glow, 1));
  geom.setAttribute("dim", new THREE.BufferAttribute(dim, 1));
  const mat = new THREE.ShaderMaterial({
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    uniforms: { scale: { value: 1 }, focus: { value: 0 } },
    vertexShader: `attribute vec3 color; attribute float glow; attribute float dim; varying vec3 vC; varying float vA;
      uniform float scale; uniform float focus;
      void main(){ vec4 mv = modelViewMatrix * vec4(position,1.0);
        float g = clamp(glow,0.0,1.0);
        vC = mix(color*0.55, vec3(1.0,0.95,0.8), g);
        float hi = (dim < 0.5 ? focus : 0.0);
        vC = mix(vC, vec3(1.0,0.85,0.45), hi*0.6);
        vA = (dim > 0.5 ? 0.05 : 0.55 + hi*0.4) + g*0.45;
        gl_PointSize = (2.2 + g*5.0 + hi*5.0) * scale / -mv.z; gl_Position = projectionMatrix * mv; }`,
    fragmentShader: `varying vec3 vC; varying float vA;
      void main(){ vec2 d = gl_PointCoord-0.5; float r = dot(d,d); if(r>0.25) discard;
        gl_FragColor = vec4(vC, vA*(1.0-r*3.0)); }`,
  });
  points = new THREE.Points(geom, mat);
  scene.add(points);
  addEventListener("resize", resize); resize();
  // click to inspect
  const ray = new THREE.Raycaster(); ray.params.Points.threshold = 0.006;
  let down;
  canvas.addEventListener("pointerdown", (e) => (down = [e.clientX, e.clientY]));
  canvas.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const r = canvas.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
    const hit = ray.intersectObject(points).find((h) => !hidden.has(groupOf[h.index]) && dim[h.index] < 0.5);
    if (hit) { showNeuron(hit.index); tab("explore"); }
  });
}

function resize() {
  const el = $("#stage");
  renderer.setSize(el.clientWidth, el.clientHeight, false);
  camera.aspect = el.clientWidth / el.clientHeight; camera.updateProjectionMatrix();
  points.material.uniforms.scale.value = el.clientHeight / 400;
}

function animate() {
  requestAnimationFrame(animate);
  let changed = false;
  for (let i = 0; i < N; i++) if (glow[i] > 0.01) { glow[i] *= 0.93; changed = true; } else if (glow[i]) { glow[i] = 0; changed = true; }
  if (changed) geom.attributes.glow.needsUpdate = true;
  controls.update();
  renderer.render(scene, camera);
}

function flash(indices, amount = 1) { for (const i of indices) glow[i] = Math.max(glow[i], amount); geom.attributes.glow.needsUpdate = true; }

function highlight(indices) {
  const all = indices === "all" || !indices || !indices.length;
  if (all) { dim.fill(0); }
  else { dim.fill(1); for (const i of indices) dim[i] = 0; }
  points.material.uniforms.focus.value = !all && indices.length < 30000 ? 1 : 0;
  for (let i = 0; i < N; i++) if (hidden.has(groupOf[i])) dim[i] = 1;
  geom.attributes.dim.needsUpdate = true;
}

function playFrames(frames, frameMs) {
  let k = 0;
  const t = setInterval(() => { if (k >= frames.length) return clearInterval(t); flash(frames[k++], 1); }, Math.max(30, frameMs * 3));
}

function initLegend() {
  $("#legend").innerHTML = GROUPS.map((g, k) => `<label><input type="checkbox" checked data-k="${k}"><i style="background:${g.color}"></i>${g.name}</label>`).join("");
  $("#legend").addEventListener("change", (e) => {
    const k = +e.target.dataset.k; e.target.checked ? hidden.delete(k) : hidden.add(k); highlight("all");
  });
}

// ---------------------------------------------------------------- the live fly
const SENSE_LABEL = { smell: "👃 Smell", taste: "👅 Taste", loom: "🖐️ Shadow", hearing: "🔊 Sound",
  wind: "💨 Wind", touch: "🪶 Touch", heat: "🔥 Heat", humidity: "💧 Humidity" };
function initSenses() {
  $("#senses").innerHTML = Object.keys(info.senses).map((s) => `<button data-s="${s}" title="${info.senses[s]}">${SENSE_LABEL[s] || s}</button>`).join("");
  $("#senses").addEventListener("click", (e) => { const s = e.target.closest("button")?.dataset.s; if (s) api("/api/poke", { sense: s }); });
}

const BEHAVIOUR_TEXT = { rest: "resting", escape: "ESCAPING! (Giant Fiber firing)", backward: "walking backward (MDN)",
  forward: "walking forward (DNp09)", turn: "turning" };
let lastSeq = -1;
async function live() {
  try {
    const s = await api("/api/live");
    if (s.seq !== lastSeq) { lastSeq = s.seq; flash(s.active, 0.8); }
    const beh = s.behaviour + (s.turn_side ? `-${s.turn_side}` : "");
    $("#fly").setAttribute("class", beh);
    const txt = s.behaviour === "turn" ? `turning ${s.turn_side} (DNa01/DNa02)` : BEHAVIOUR_TEXT[s.behaviour];
    $("#behaviour").textContent = s.paused ? "asleep (no one watching)" : txt;
    $("#behaviour").className = `pill ${s.behaviour}`;
    $("#fly-caption").textContent = s.feeling ? `feeling ${info.senses[s.feeling]}` : `${s.active_count.toLocaleString()} neurons fired`;
    $("#brainclock").textContent = `brain time ${(s.brain_ms / 1000).toFixed(1)} s`;
    document.querySelectorAll("#senses button").forEach((b) => b.classList.toggle("feeling", b.dataset.s === s.feeling));
  } catch (e) { $("#behaviour").textContent = "brain offline"; }
  setTimeout(live, 300);
}

// ---------------------------------------------------------------- tabs
function tab(name) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.id === `tab-${name}`));
}
function initTabs() { $("#tabs").addEventListener("click", (e) => e.target.dataset.tab && tab(e.target.dataset.tab)); }

// ---------------------------------------------------------------- talk
const SUGGEST = ["What are you doing?", "Here comes a shadow!", "Tell me about your giant fiber",
  "What happens when you smell something?", "How many neurons do you have?", "What is GABA?",
  "What is your mushroom body?", "How do you compare to a human brain?", "How does this simulation work?"];
function say(text, who) {
  const d = document.createElement("div"); d.className = `msg ${who}`; d.textContent = text;
  $("#chat").appendChild(d); $("#chat").scrollTop = 1e9; return d;
}
function suggest() {
  const pick = SUGGEST.slice().sort(() => Math.random() - 0.5).slice(0, 3);
  $("#suggest").innerHTML = pick.map((s) => `<button>${s}</button>`).join("");
}
async function ask(text) {
  say(text, "me");
  suggest();
  const wait = say("…thinking with 165,122 neurons…", "fly");
  const r = await api("/api/chat", { text });
  wait.textContent = r.reply;
  if (r.highlight?.length) { highlight(r.highlight); setTimeout(() => highlight("all"), 12000); }
  if (r.experiment) playFrames(r.experiment.frames, r.experiment.frame_ms);
}
function initChat() {
  say("Hi! I'm a fruit fly. My whole brain is running on this computer right now, and it decides what I do. Ask me anything, or press a sense button below to make me feel something.", "fly");
  $("#suggest").innerHTML = SUGGEST.slice(0, 5).map((s) => `<button>${s}</button>`).join("");
  $("#suggest").addEventListener("click", (e) => e.target.tagName === "BUTTON" && ask(e.target.textContent));
  $("#ask").addEventListener("submit", (e) => { e.preventDefault(); const q = $("#q").value.trim(); if (q) { $("#q").value = ""; ask(q); } });
}

// ---------------------------------------------------------------- learn
const done = new Set(JSON.parse((() => { try { return localStorage.getItem("fruitos-done") || "[]"; } catch { return "[]"; } })()));
function initLessons() {
  const render = () => {
    $("#lesson-list").innerHTML = info.lessons.map((l) => `<button data-id="${l.id}" class="${done.has(l.id) ? "done" : ""}">${l.title}</button>`).join("");
  };
  render();
  $("#lesson-list").addEventListener("click", async (e) => {
    const id = e.target.dataset.id; if (!id) return;
    const l = await api(`/api/lesson?id=${id}`);
    highlight(l.highlight);
    if (l.action.poke) api("/api/poke", { sense: l.action.poke });
    const qz = l.quiz;
    $("#lesson").innerHTML = `<h3>${l.title}</h3><p>${l.text}</p>${l.action.poke ? `<p class="muted">Watch the brain: I just gave the fly a sensation.</p>` : ""}
      <div class="quiz"><b>Quick check:</b> ${qz.question}${qz.choices.map((c) => `<button>${c}</button>`).join("")}</div>`;
    $("#lesson .quiz").addEventListener("click", (ev) => {
      if (ev.target.tagName !== "BUTTON") return;
      const right = ev.target.textContent === qz.answer;
      ev.target.classList.add(right ? "right" : "wrong");
      if (right) { done.add(id); try { localStorage.setItem("fruitos-done", JSON.stringify([...done])); } catch {} render(); }
    });
  });
}

// ---------------------------------------------------------------- explore
function initExplore() {
  let t;
  $("#search").addEventListener("input", () => { clearTimeout(t); t = setTimeout(async () => {
    const res = await api(`/api/search?q=${encodeURIComponent($("#search").value)}`);
    $("#results").innerHTML = res.map((r, k) => `<button class="chip" data-k="${k}">${r.type} ×${r.count}</button>`).join("");
    $("#results").onclick = (e) => { const r = res[e.target.dataset.k]; if (r) { highlight(r.indices); flash(r.indices); showNeuron(r.indices[0]); } };
  }, 200); });
}
async function showNeuron(i) {
  const n = await api(`/api/neuron?i=${i}`);
  flash([i]);
  const link = (p) => `<a data-i="${p.index}">${p.type}</a> (${p.synapses})`;
  $("#neuron").innerHTML = `<h3>${n.type || "untyped neuron"} <span class="muted">${n.instance}</span></h3>
    <table>
      <tr><td>Body ID</td><td>${n.bodyId}</td></tr>
      <tr><td>Part of</td><td>${n.superclass.replaceAll("_", " ")} · ${n.class.replaceAll("_", " ")}</td></tr>
      <tr><td>Transmitter</td><td>${n.transmitter}</td></tr>
      <tr><td>Side</td><td>${n.side}</td></tr>
      <tr><td>Position (µm)</td><td>${n.position_um.join(", ")}${n.position_measured ? "" : " (estimated)"}</td></tr>
      <tr><td>Inputs</td><td>${n.inputs.toLocaleString()} neurons, ${n.input_synapses.toLocaleString()} synapses</td></tr>
      <tr><td>Outputs</td><td>${n.outputs.toLocaleString()} neurons, ${n.output_synapses.toLocaleString()} synapses</td></tr>
      <tr><td>Strongest inputs</td><td>${n.top_inputs.map(link).join(", ")}</td></tr>
      <tr><td>Strongest outputs</td><td>${n.top_outputs.map(link).join(", ")}</td></tr>
    </table>
    <p><button class="chip" id="ask-about">Ask the fly about ${n.type || "this neuron"}</button></p>`;
  $("#neuron").onclick = (e) => {
    if (e.target.dataset.i) showNeuron(+e.target.dataset.i);
    if (e.target.id === "ask-about" && n.type) { tab("talk"); ask(`Tell me about ${n.type}`); }
  };
  highlight([i, ...n.top_inputs.map((p) => p.index), ...n.top_outputs.map((p) => p.index)]);
}

boot();
