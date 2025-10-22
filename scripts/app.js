const ovLib = document.getElementById('overlay-lib');
const ovRAG = document.getElementById('overlay-rag');
const MiniSearchLib = window.MiniSearch;
const idbKeyvalLib = window.idbKeyval;

if (!MiniSearchLib) {
  throw new Error('MiniSearch library failed to load');
}
if (!idbKeyvalLib) {
  throw new Error('idb-keyval library failed to load');
}

function show(el) {
  el.style.display = 'flex';
}

function hide(el) {
  el.style.display = 'none';
}

function setOverlayMessage(el, message) {
  const span = el.querySelector('.message');
  if (span) {
    span.textContent = message;
  }
}

show(ovLib);
setOverlayMessage(ovLib, 'Loading Transformers.js library…');

async function loadTransformers() {
  const versions = ['2.17.1', '2.16.1', '2.12.0'];
  const cdns = [
    (v) => `https://cdn.jsdelivr.net/npm/@xenova/transformers@${v}/dist/transformers.min.js`,
  ];
  let lastErr = null;
  for (const v of versions) {
    for (const gen of cdns) {
      const url = gen(v);
      try {
        setOverlayMessage(ovLib, `Loading Transformers.js ${v}…`);
        console.log('[loader] trying', url);
        const mod = await import(/* @vite-ignore */ url);
        if (!mod.pipeline) {
          throw new Error('pipeline missing in module exports');
        }
        console.log('[loader] success', url, 'exports=', Object.keys(mod));
        mod.__TRANSFORMERS_CDN__ = url;
        mod.__TRANSFORMERS_VER__ = v;
        hide(ovLib);
        return mod;
      } catch (e) {
        console.warn('[loader] failed', url, e);
        lastErr = e;
        setOverlayMessage(ovLib, `Failed ${v}, trying fallback…`);
      }
    }
  }
  hide(ovLib);
  throw lastErr || new Error('Unable to load @xenova/transformers from all CDNs/versions');
}

const { pipeline, env, version, __TRANSFORMERS_CDN__, __TRANSFORMERS_VER__ } = await loadTransformers();

const tabs = document.querySelectorAll('.tab');
const panes = {
  logs: document.getElementById('pane-logs'),
  traces: document.getElementById('pane-traces'),
  info: document.getElementById('pane-info'),
};
tabs.forEach((t) => (t.onclick = () => {
  tabs.forEach((x) => x.classList.remove('active'));
  t.classList.add('active');
  const name = t.dataset.tab;
  Object.entries(panes).forEach(([k, el]) => el.classList.toggle('active', k === name));
}));

const logWrap = document.getElementById('diag-log');
function logLine(...args) {
  console.log(...args);
  const div = document.createElement('div');
  div.className = 'logline';
  const ts = document.createElement('span');
  ts.className = 'ts';
  ts.textContent = new Date().toLocaleTimeString() + ' —';
  div.appendChild(ts);
  div.appendChild(
    document.createTextNode(
      args
        .map((a) => {
          try {
            return typeof a === 'object' ? JSON.stringify(a) : String(a);
          } catch (err) {
            return String(a);
          }
        })
        .join(' '),
    ),
  );
  logWrap.appendChild(div);
  logWrap.scrollTop = logWrap.scrollHeight;
}

const hasGPU = !!navigator.gpu;
document.getElementById('gpu-badge').textContent = 'GPU: ' + (hasGPU ? 'WebGPU available' : 'No WebGPU');
document.getElementById('version-badge').textContent =
  'Transformers.js: ' + (__TRANSFORMERS_VER__ || version || 'unknown');
document.getElementById('cdnInfo').textContent = 'CDN: ' + __TRANSFORMERS_CDN__;
document.getElementById('gpuInfo').textContent = 'GPU: ' + (hasGPU ? 'WebGPU available' : 'No WebGPU');

try {
  env.allowLocalModels = false;
  env.backends.onnx.preferWebGPU = hasGPU;
  env.backends.onnx.wasm.proxy = true;
  env.backends.onnx.wasm.numThreads = 1;
  logLine('[env] backends =', env.backends);
  document.getElementById('backend-badge').textContent = 'Backend: ' + (hasGPU ? 'WebGPU/ONNX' : 'WASM/ONNX');
} catch (e) {
  logLine('[env] backend config error:', e);
  document.getElementById('backend-badge').textContent = 'Backend: config error';
}

const chatlog = document.getElementById('chatlog');
const q = document.getElementById('q');
const btn = document.getElementById('send');
const tracesBody = document.querySelector('#traces tbody');
const inTopBM = document.getElementById('inTopBM');
const inTopK = document.getElementById('inTopK');
const inBMTh = document.getElementById('inBMTh');
const btnCheckImport = document.getElementById('btn-check-import');
const btnSelfTest = document.getElementById('btn-self-test');
const selGen = document.getElementById('selGen');
const selEmb = document.getElementById('selEmb');
const btnApply = document.getElementById('btn-apply');
const btnClearCurrent = document.getElementById('btn-clear-current');
const btnClearAll = document.getElementById('btn-clear-all');
const pillGen = document.getElementById('pillGen');
const pillEmb = document.getElementById('pillEmb');

function addBubble(role, html) {
  const row = document.createElement('div');
  row.className = `row ${role}`;
  const b = document.createElement('div');
  b.className = 'bubble';
  b.innerHTML = html;
  row.appendChild(b);
  chatlog.appendChild(row);
  chatlog.scrollTop = chatlog.scrollHeight;
  return b;
}

const JSON_FILES = ['data.json', 'private_knowledge.json'];

async function sha256OfText(text) {
  const enc = new TextEncoder().encode(text);
  const buf = await crypto.subtle.digest('SHA-256', enc);
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}
function norm(s) {
  return (s || '').replace(/\s+/g, ' ').trim();
}
function walkJSON(obj, path = '', out = []) {
  if (obj == null) return out;
  if (typeof obj === 'string') {
    const t = norm(obj);
    if (t) out.push({ section: path || 'root', text: t });
    return out;
  }
  if (Array.isArray(obj)) {
    obj.forEach((v) => walkJSON(v, path, out));
    return out;
  }
  if (typeof obj === 'object') {
    for (const [k, v] of Object.entries(obj)) {
      const p = path ? `${path}.${k}` : k;
      if (typeof v === 'string') {
        const t = norm(v);
        if (t) out.push({ section: p, text: t });
      } else {
        walkJSON(v, p, out);
      }
    }
  }
  return out;
}
function chunkText(text, size = 900, overlap = 150) {
  if (text.length <= size) return [text];
  let i = 0;
  const res = [];
  while (i < text.length) {
    const end = Math.min(text.length, i + size);
    res.push(text.slice(i, end));
    if (end === text.length) break;
    i = Math.max(0, end - overlap);
  }
  return res;
}

let CHUNKS = [];
let mini = null;

const t0Load = performance.now();
async function buildIndex() {
  logLine('[data] start loading JSON files:', JSON_FILES);
  setOverlayMessage(ovRAG, 'Loading JSON data…');
  const contents = await Promise.all(
    JSON_FILES.map(async (src) => {
      const r = await fetch(src);
      const t = await r.text();
      logLine(`[data] ${src} bytes=`, t.length);
      return { src, text: t, json: JSON.parse(t) };
    }),
  );

  const sectionMap = new Map();
  for (const { src, json } of contents) {
    const pairs = walkJSON(json);
    for (const { section, text } of pairs) {
      const key = `${src}::${section || 'root'}`;
      if (!sectionMap.has(key)) sectionMap.set(key, { src, section, items: [] });
      sectionMap.get(key).items.push(text);
    }
  }

  CHUNKS = [];
  let id = 0;
  for (const { src, section, items } of sectionMap.values()) {
    const joined = norm(items.join(' '));
    const parts = chunkText(joined);
    parts.forEach((ch, i) => {
      if (ch.length >= 20) {
        CHUNKS.push({ id: (id++).toString(), text: ch, section, source: src, chunkId: `${section}#${i}` });
      }
    });
  }
  logLine('[data] total sections:', sectionMap.size, 'chunks:', CHUNKS.length);

  mini = new MiniSearchLib({
    fields: ['text'],
    storeFields: ['text', 'section', 'source', 'chunkId'],
    searchOptions: { boost: { text: 1 } },
  });
  mini.addAll(CHUNKS);
  const t1 = performance.now();
  logLine(`[data] index built in ${(t1 - t0Load).toFixed(1)} ms`);
}

const LS = {
  gen: 'rag.gen.model',
  emb: 'rag.emb.model',
  topBM: 'rag.topBM',
  topK: 'rag.topK',
  bmTh: 'rag.bmTh',
};
function loadLS() {
  if (localStorage.getItem(LS.gen)) selGen.value = localStorage.getItem(LS.gen);
  if (localStorage.getItem(LS.emb)) selEmb.value = localStorage.getItem(LS.emb);
  if (localStorage.getItem(LS.topBM)) inTopBM.value = localStorage.getItem(LS.topBM);
  if (localStorage.getItem(LS.topK)) inTopK.value = localStorage.getItem(LS.topK);
  if (localStorage.getItem(LS.bmTh)) inBMTh.value = localStorage.getItem(LS.bmTh);
}
function saveLS() {
  localStorage.setItem(LS.gen, selGen.value);
  localStorage.setItem(LS.emb, selEmb.value);
  localStorage.setItem(LS.topBM, inTopBM.value);
  localStorage.setItem(LS.topK, inTopK.value);
  localStorage.setItem(LS.bmTh, inBMTh.value);
}
loadLS();
pillGen.textContent = 'Generator: ' + selGen.value.split('/').pop();
pillEmb.textContent = 'Embedder: ' + selEmb.value.split('/').pop();

let EMBED_MODEL = selEmb.value;
let GEN_MODEL = selGen.value;

async function computeSignature() {
  const settings = `${EMBED_MODEL}|${GEN_MODEL}|size=900|overlap=150`;
  const fileHashes = await Promise.all(
    JSON_FILES.map(async (src) => {
      const r = await fetch(src);
      const t = await r.text();
      return `${src}:${await sha256OfText(t)}`;
    }),
  );
  const sig = await sha256OfText(settings + '|' + fileHashes.join('|'));
  logLine('[cache] signature:', sig, 'settings:', settings);
  return sig;
}

let embedder = null;
let generator = null;
let DOC_EMB = null;

async function loadPipelines() {
  logLine('[pipelines] loading pipelines…', { EMBED_MODEL, GEN_MODEL, cdn: __TRANSFORMERS_CDN__ });
  setOverlayMessage(ovRAG, 'Loading Transformers pipelines…');
  const t0 = performance.now();
  embedder = await pipeline('feature-extraction', EMBED_MODEL, { quantized: true });
  generator = await pipeline('text2text-generation', GEN_MODEL, { quantized: true });
  const t1 = performance.now();
  logLine(`[pipelines] loaded in ${(t1 - t0).toFixed(1)} ms`);
}
async function embedTextMeanNorm(text) {
  const out = await embedder(text, { pooling: 'mean', normalize: true });
  return new Float32Array(out);
}
async function loadOrComputeDocEmbeddings() {
  setOverlayMessage(ovRAG, 'Preparing document embeddings…');
  const sig = await computeSignature();
  const cacheKey = `embeds::${sig}`;
  logLine('[cache] try load:', cacheKey);

  try {
    const cached = await idbKeyvalLib.get(cacheKey);
    if (cached && Array.isArray(cached) && cached.length === CHUNKS.length) {
      setOverlayMessage(ovRAG, 'Loaded cached embeddings from IndexedDB.');
      DOC_EMB = cached.map((arr) => new Float32Array(arr));
      logLine('[cache] HIT — loaded embeddings from IndexedDB:', DOC_EMB.length);
      return { cacheKey, hit: true };
    }
    logLine('[cache] MISS — no cache or size mismatch.');
  } catch (e) {
    logLine('[cache] load error:', e);
  }

  logLine('[embed] computing doc embeddings… count:', CHUNKS.length);
  const t0 = performance.now();
  DOC_EMB = new Array(CHUNKS.length);
  for (let i = 0; i < CHUNKS.length; i++) {
    const e0 = performance.now();
    DOC_EMB[i] = await embedTextMeanNorm(CHUNKS[i].text);
    const e1 = performance.now();
    if (i % 20 === 0) {
      const progress = `${i + 1}/${CHUNKS.length}`;
      setOverlayMessage(ovRAG, `Embedding documents… ${progress}`);
      logLine(`[embed] progress: ${i + 1}/${CHUNKS.length}  (+${(e1 - e0).toFixed(1)} ms)`);
    }
  }
  const t1 = performance.now();
  logLine(`[embed] done in ${(t1 - t0).toFixed(1)} ms`);

  const dump = DOC_EMB.map((v) => Array.from(v));
  try {
    await idbKeyvalLib.set(cacheKey, dump);
    logLine('[cache] saved embeddings to IndexedDB:', cacheKey);
  } catch (e) {
    logLine('[cache] save error (quota or structured clone):', e);
  }
  setOverlayMessage(ovRAG, 'Embeddings ready.');
  return { cacheKey, hit: false };
}

show(ovRAG);
await buildIndex();
await loadPipelines();
await loadOrComputeDocEmbeddings();
hide(ovRAG);
logLine('[init] RAG stack ready.');

function cosineSim(a, b) {
  let s = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    s += x * y;
    na += x * x;
    nb += y * y;
  }
  return s / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
}
async function retrieve(query, kBM = 12, k = 4, bmTh = 0.1) {
  const t0 = performance.now();
  const bm = mini.search(query, { fuzzy: 0.1, prefix: true });
  const bmBest = bm[0]?.score || 0;
  const useGlobal = bmBest < bmTh;

  const qv = await embedTextMeanNorm(query);
  let top = [];
  if (useGlobal) {
    logLine(`[retrieval] BM25 weak (best=${bmBest.toFixed(4)} < th=${bmTh}) → global cosine`);
    const scored = CHUNKS.map((d, i) => ({ i, cos: cosineSim(qv, DOC_EMB[i]) }));
    scored.sort((a, b) => b.cos - a.cos);
    top = scored.slice(0, k).map((s) => ({ ...CHUNKS[s.i], bm25: 0, cos: s.cos }));
  } else {
    logLine(`[retrieval] BM25 ok (best=${bmBest.toFixed(4)} ≥ th=${bmTh}) → rerank ${Math.min(kBM, bm.length)} cands by cosine`);
    const cand = bm.slice(0, kBM).map((r) => ({ i: parseInt(r.id, 10), bm: r.score }));
    const rescored = cand.map((c) => ({ ...c, cos: cosineSim(qv, DOC_EMB[c.i]) }));
    rescored.sort((a, b) => b.cos - a.cos);
    top = rescored.slice(0, k).map((s) => ({ ...CHUNKS[s.i], bm25: s.bm, cos: s.cos }));
  }
  const t1 = performance.now();
  logLine(`[retrieval] found=${top.length} in ${(t1 - t0).toFixed(1)} ms`);
  return top;
}
function renderTraces(items) {
  tracesBody.innerHTML = '';
  items.forEach((it, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${idx + 1}</td><td>${it.section}</td><td>${it.source.replace('./', '')}</td><td>${it.chunkId}</td><td>${(it.bm25 ?? 0).toFixed(4)}</td><td>${(it.cos ?? 0).toFixed(4)}</td>`;
    tracesBody.appendChild(tr);
  });
}

async function generateVi(query, contexts) {
  const ctx = contexts.map((c, i) => `[${i + 1}] ${c.text}`).join(' ');
  const prompt = `Hãy trả lời ngắn gọn, rõ ràng và tự nhiên bằng tiếng Việt dựa trên NGỮ CẢNH sau: ${ctx}\n\nCÂU HỎI: ${query}\n\nTrả lời bằng tiếng Việt, có thể tham chiếu [1], [2]... nếu cần.`;
  logLine('[gen] prompt.len=', prompt.length);

  const t0 = performance.now();
  const out = await generator(prompt, { max_new_tokens: 200 });
  const t1 = performance.now();
  const text = out?.[0]?.generated_text || '(không có kết quả)';
  logLine(`[gen] generated in ${(t1 - t0).toFixed(1)} ms`, { preview: text.slice(0, 120) + '…' });

  const bot = addBubble('bot', '');
  const parts = text.split(/(\s+)/);
  for (const p of parts) {
    bot.textContent += p;
    chatlog.scrollTop = chatlog.scrollHeight;
    await new Promise((r) => setTimeout(r, 8));
  }
  return text;
}

function parseNum(el, def) {
  const v = parseFloat(el.value);
  return Number.isFinite(v) ? v : def;
}
async function onAsk() {
  const query = q.value.trim();
  const topBM = parseNum(inTopBM, 12);
  const topK = parseNum(inTopK, 4);
  const bmTh = parseNum(inBMTh, 0.1);
  if (!query) return;

  addBubble('user', query);
  q.value = '';

  saveLS();
  logLine('[ask] query:', query, { topBM, topK, bmTh, GEN_MODEL, EMBED_MODEL });

  const top = await retrieve(query, topBM, topK, bmTh);
  renderTraces(top);
  await generateVi(query, top);
}
btn.onclick = onAsk;
q.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) onAsk();
});

btnCheckImport.onclick = async () => {
  try {
    const keys = Object.keys(await import(/* @vite-ignore */ __TRANSFORMERS_CDN__));
    logLine('[diagnostics] import OK. exports keys:', keys);
    alert('Import OK. Check Logs tab.');
  } catch (e) {
    logLine('[diagnostics] import FAIL:', e);
    alert('Import FAIL. Xem Logs tab.');
  }
};
btnSelfTest.onclick = async () => {
  try {
    logLine('[self-test] starting text-classification (sentiment)...');
    const sst = await pipeline('text-classification', 'Xenova/distilbert-base-uncased-finetuned-sst-2-english', { quantized: true });
    const out = await sst('Transformers.js is working!');
    logLine('[self-test] output:', out);
    alert('Self-test OK. Xem Logs tab.');
  } catch (e) {
    logLine('[self-test] FAIL:', e);
    alert('Self-test FAIL. Xem Logs tab.');
  }
};

btnApply.onclick = async () => {
  EMBED_MODEL = selEmb.value;
  GEN_MODEL = selGen.value;
  pillGen.textContent = 'Generator: ' + GEN_MODEL.split('/').pop();
  pillEmb.textContent = 'Embedder: ' + EMBED_MODEL.split('/').pop();
  saveLS();

  show(ovRAG);
  setOverlayMessage(ovRAG, 'Applying selected models…');
  logLine('[models] applying…', { EMBED_MODEL, GEN_MODEL });
  await loadPipelines();
  await loadOrComputeDocEmbeddings();
  hide(ovRAG);
  alert('Models applied. Embeddings loaded (cache or recomputed).');
};

btnClearCurrent.onclick = async () => {
  const sig = await computeSignature();
  const cacheKey = `embeds::${sig}`;
  try {
    await idbKeyvalLib.del(cacheKey);
    logLine('[cache] cleared current:', cacheKey);
    alert('Cleared current cache.');
  } catch (e) {
    logLine('[cache] clear current error:', e);
    alert('Error clearing current cache (xem Logs).');
  }
};

btnClearAll.onclick = async () => {
  try {
    const keys = await idbKeyvalLib.keys();
    let count = 0;
    for (const k of keys) {
      if (typeof k === 'string' && k.startsWith('embeds::')) {
        await idbKeyvalLib.del(k);
        count++;
      }
    }
    logLine(`[cache] cleared ALL embeds::* (${count} keys).`);
    alert(`Cleared ALL embeds::* (${count}).`);
  } catch (e) {
    logLine('[cache] clear all error:', e);
    alert('Error clearing all cache (xem Logs).');
  }
};
