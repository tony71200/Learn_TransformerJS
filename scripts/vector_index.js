const CDN_CANDIDATES = [
  'https://cdn.jsdelivr.net/npm/@unum-cloud/usearch@2.12.0/wasm/usearch.min.js',
  'https://unpkg.com/@unum-cloud/usearch@2.12.0/wasm/usearch.min.js',
];

async function loadUSearchModule() {
  for (const url of CDN_CANDIDATES) {
    try {
      console.log('[usearch] trying', url);
      const mod = await import(/* @vite-ignore */ url);
      if (mod && (mod.default || mod.Index)) {
        return mod.default ?? mod;
      }
    } catch (err) {
      console.warn('[usearch] failed', url, err);
    }
  }
  return null;
}

function cosine(a, b) {
  let sum = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    sum += x * y;
    na += x * x;
    nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb) + 1e-12;
  return sum / denom;
}

export class VectorIndex {
  constructor(dim, metric = 'cos') {
    this.dim = dim;
    this.metric = metric;
    this.backend = 'flat';
    this.usearch = null;
    this.idx = null;
    this.vectors = [];
    this.ids = [];
    this.map = new Map();
    this.ready = false;
  }

  async init() {
    const lib = await loadUSearchModule();
    if (lib && lib.Index) {
      this.usearch = lib;
      try {
        this.idx = new lib.Index({ metric: this.metric, dimensions: this.dim });
        this.backend = 'usearch';
        console.log('[VectorIndex] backend=usearch');
      } catch (err) {
        console.warn('[VectorIndex] failed to init USearch index, fallback to flat', err);
        this.idx = null;
        this.backend = 'flat';
      }
    } else {
      console.log('[VectorIndex] backend=flat');
    }
    this.ready = false;
  }

  setFallbackPairs(pairs) {
    this.ids = pairs.map((p) => String(p.id));
    this.vectors = pairs.map((p) => p.vec);
    this.map = new Map(this.ids.map((id, i) => [id, this.vectors[i]]));
  }

  async rebuild(pairs) {
    this.setFallbackPairs(pairs);
    if (this.backend === 'usearch' && this.usearch) {
      this.idx = new this.usearch.Index({ metric: this.metric, dimensions: this.dim });
      for (const { id, vec } of pairs) {
        await this.idx.add(Number(id), vec);
      }
    }
    this.ready = true;
  }

  async load(buffer) {
    if (this.backend !== 'usearch' || !buffer) {
      return false;
    }
    if (!(buffer instanceof ArrayBuffer)) {
      if (buffer.buffer instanceof ArrayBuffer) {
        buffer = buffer.buffer;
      } else {
        buffer = buffer.slice(0).buffer ?? buffer;
      }
    }
    this.idx = new this.usearch.Index({ metric: this.metric, dimensions: this.dim });
    await this.idx.load(buffer);
    this.ready = true;
    return true;
  }

  async add(id, vec) {
    if (this.backend === 'usearch' && this.idx) {
      await this.idx.add(Number(id), vec);
    }
    this.map.set(String(id), vec);
    this.ids.push(String(id));
    this.vectors.push(vec);
  }

  async addBulk(pairs) {
    for (const p of pairs) {
      await this.add(p.id, p.vec);
    }
  }

  async search(queryVec, k = 10) {
    if (!this.ready) {
      throw new Error('VectorIndex not ready');
    }
    if (this.backend === 'usearch' && this.idx) {
      const res = await this.idx.search(queryVec, k);
      const keys = Array.from(res.keys || []);
      const distances = res.distances ? Array.from(res.distances) : [];
      return keys.map((key, i) => ({ id: String(key), score: 1 - (distances[i] ?? 0) }));
    }
    const sims = this.vectors.map((vec, i) => ({ id: this.ids[i], score: cosine(queryVec, vec) }));
    sims.sort((a, b) => b.score - a.score);
    return sims.slice(0, k);
  }

  get size() {
    return this.ids.length;
  }

  async save() {
    if (this.backend === 'usearch' && this.idx) {
      return this.idx.save();
    }
    return null;
  }
}
