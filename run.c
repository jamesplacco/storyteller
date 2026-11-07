// run.c: the Storyteller GPT forward pass and sampler in plain C (Chapter 8).
// No libraries beyond libc. Same math as model.py, one token at a time, with a KV-cache.
//
// build:  clang -O3 -ffast-math -o run run.c -lm        (Linux: gcc ... -lm)
// usage:  ./run model.bin tok.bin [-t temp] [-n tokens] [-s seed] [-i "prompt"] [-c "chat request"]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

typedef struct { int n_layer, n_head, n_embd, block_size, vocab_size; } Config;

typedef struct {   // pointers into one big float buffer; per-layer tensors are stacked
    float *wte, *wpe;
    float *ln1w, *ln1b, *qkvw, *qkvb, *projw, *projb, *ln2w, *ln2b, *fcw, *fcb, *fcpw, *fcpb;
    float *lnfw, *lnfb;
} Weights;

typedef struct {   // activations for the current token, plus the KV-cache
    float *x, *xb, *xb2, *qkv, *hb, *att, *logits, *kcache, *vcache;
} State;

static Config cfg;
static Weights w;
static State s;

static float *load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int h[8];
    if (fread(h, sizeof(int), 8, f) != 8 || h[0] != 0x53544c52) { fprintf(stderr, "%s is not a model.bin\n", path); exit(1); }
    cfg = (Config){h[2], h[3], h[4], h[5], h[6]};
    fseek(f, 0, SEEK_END);
    long nfloat = (ftell(f) - 8 * sizeof(int)) / sizeof(float);
    fseek(f, 8 * sizeof(int), SEEK_SET);
    float *buf = malloc(nfloat * sizeof(float));
    if (fread(buf, sizeof(float), nfloat, f) != (size_t)nfloat) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);

    int L = cfg.n_layer, C = cfg.n_embd;
    float *p = buf;
    #define TAKE(dst, n) do { dst = p; p += (long)(n); } while (0)
    TAKE(w.wte, (long)cfg.vocab_size * C); TAKE(w.wpe, (long)cfg.block_size * C);
    TAKE(w.ln1w, L * C); TAKE(w.ln1b, L * C);
    TAKE(w.qkvw, (long)L * 3 * C * C); TAKE(w.qkvb, L * 3 * C);
    TAKE(w.projw, (long)L * C * C); TAKE(w.projb, L * C);
    TAKE(w.ln2w, L * C); TAKE(w.ln2b, L * C);
    TAKE(w.fcw, (long)L * 4 * C * C); TAKE(w.fcb, L * 4 * C);
    TAKE(w.fcpw, (long)L * 4 * C * C); TAKE(w.fcpb, L * C);
    TAKE(w.lnfw, C); TAKE(w.lnfb, C);
    if (p - buf != nfloat) { fprintf(stderr, "size mismatch: %ld vs %ld floats\n", (long)(p - buf), nfloat); exit(1); }

    s.x = calloc(C, 4); s.xb = calloc(C, 4); s.xb2 = calloc(C, 4); s.qkv = calloc(3 * C, 4);
    s.hb = calloc(4 * C, 4); s.att = calloc(cfg.block_size, 4); s.logits = calloc(cfg.vocab_size, 4);
    s.kcache = calloc((long)L * cfg.block_size * C, 4); s.vcache = calloc((long)L * cfg.block_size * C, 4);
    return buf;
}

// ---------------- the building blocks ----------------
static void layernorm(float *out, const float *x, const float *g, const float *b, int n) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    float rs = 1.0f / sqrtf(var / n + 1e-5f);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * rs * g[i] + b[i];
}

// out[d] = W[d, :] . x + b[d]; W is (d_out, d_in) row-major, exactly as nn.Linear stores it
static void matmul(float *out, const float *x, const float *W, const float *b, int d_in, int d_out) {
    for (int i = 0; i < d_out; i++) {
        const float *row = W + (long)i * d_in;
        float acc = b ? b[i] : 0.0f;
        for (int j = 0; j < d_in; j++) acc += row[j] * x[j];
        out[i] = acc;
    }
}

static float gelu(float x) {   // tanh approximation, as in GPT-2
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void softmax(float *x, int n) {
    float m = x[0], sum = 0;
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ---------------- one token through the whole model ----------------
static float *forward(int token, int pos) {
    int C = cfg.n_embd, H = cfg.n_head, hs = C / H, T = cfg.block_size;
    for (int i = 0; i < C; i++) s.x[i] = w.wte[(long)token * C + i] + w.wpe[(long)pos * C + i];

    for (int l = 0; l < cfg.n_layer; l++) {
        // attention: communicate
        layernorm(s.xb, s.x, w.ln1w + l * C, w.ln1b + l * C, C);
        matmul(s.qkv, s.xb, w.qkvw + (long)l * 3 * C * C, w.qkvb + l * 3 * C, C, 3 * C);
        float *q = s.qkv, *kc = s.kcache + (long)l * T * C, *vc = s.vcache + (long)l * T * C;
        memcpy(kc + (long)pos * C, s.qkv + C, C * sizeof(float));       // append this token's k, v
        memcpy(vc + (long)pos * C, s.qkv + 2 * C, C * sizeof(float));
        for (int h = 0; h < H; h++) {
            float *qh = q + h * hs, scale = 1.0f / sqrtf((float)hs);
            for (int t = 0; t <= pos; t++) {                            // causal: only past and present
                float *kh = kc + (long)t * C + h * hs, d = 0;
                for (int i = 0; i < hs; i++) d += qh[i] * kh[i];
                s.att[t] = d * scale;
            }
            softmax(s.att, pos + 1);
            float *out = s.xb2 + h * hs;
            for (int i = 0; i < hs; i++) out[i] = 0;
            for (int t = 0; t <= pos; t++) {
                float a = s.att[t], *vh = vc + (long)t * C + h * hs;
                for (int i = 0; i < hs; i++) out[i] += a * vh[i];
            }
        }
        matmul(s.xb, s.xb2, w.projw + (long)l * C * C, w.projb + l * C, C, C);
        for (int i = 0; i < C; i++) s.x[i] += s.xb[i];                // residual

        // MLP: compute
        layernorm(s.xb, s.x, w.ln2w + l * C, w.ln2b + l * C, C);
        matmul(s.hb, s.xb, w.fcw + (long)l * 4 * C * C, w.fcb + l * 4 * C, C, 4 * C);
        for (int i = 0; i < 4 * C; i++) s.hb[i] = gelu(s.hb[i]);
        matmul(s.xb, s.hb, w.fcpw + (long)l * 4 * C * C, w.fcpb + l * C, 4 * C, C);
        for (int i = 0; i < C; i++) s.x[i] += s.xb[i];                // residual
    }
    layernorm(s.x, s.x, w.lnfw, w.lnfb, C);
    matmul(s.logits, s.x, w.wte, NULL, C, cfg.vocab_size);              // tied output layer
    return s.logits;
}

// ---------------- tokenizer: decode, and a byte-level BPE encoder ----------------
static int n_tok, n_merges, EOT, USER, ASSISTANT;
static unsigned char **vocab; static int *vlen; static int *merges;   // merges[2*r], merges[2*r+1]

static void load_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int h[5];
    if (fread(h, sizeof(int), 5, f) != 5) exit(1);
    n_tok = h[0]; n_merges = h[1]; EOT = h[2]; USER = h[3]; ASSISTANT = h[4];
    vocab = malloc(n_tok * sizeof(char *)); vlen = malloc(n_tok * sizeof(int));
    for (int i = 0; i < n_tok; i++) {
        if (fread(&vlen[i], sizeof(int), 1, f) != 1) exit(1);
        vocab[i] = malloc(vlen[i] + 1);
        if (fread(vocab[i], 1, vlen[i], f) != (size_t)vlen[i]) exit(1);
    }
    merges = malloc(2 * n_merges * sizeof(int));
    if (fread(merges, sizeof(int), 2 * n_merges, f) != (size_t)(2 * n_merges)) exit(1);
    fclose(f);
}

static int merge_rank(int a, int b) {   // lower rank = learned earlier = applied first
    for (int r = 0; r < n_merges; r++) if (merges[2 * r] == a && merges[2 * r + 1] == b) return r;
    return -1;
}

static int encode_chunk(const unsigned char *c, int n, int *out) {
    for (int i = 0; i < n; i++) out[i] = c[i];
    for (;;) {
        int best = -1, at = -1;
        for (int i = 0; i + 1 < n; i++) {
            int r = merge_rank(out[i], out[i + 1]);
            if (r >= 0 && (best < 0 || r < best)) { best = r; at = i; }
        }
        if (best < 0) return n;
        out[at] = 256 + best;
        memmove(out + at + 1, out + at + 2, (n - at - 2) * sizeof(int));
        n--;
    }
}

static int cls(unsigned char ch) {  // the same character classes as the regex in bpe.py
    if (isalpha(ch)) return 1;
    if (isdigit(ch)) return 2;
    if (isspace(ch)) return 0;
    return 3;
}

static int encode(const char *text, int *out) {
    const unsigned char *t = (const unsigned char *)text;
    int n = strlen(text), i = 0, k = 0;
    while (i < n) {                    // pre-split: " ?letters", " ?digits", " ?other", or whitespace
        int j = i, c;
        if (t[i] == ' ' && i + 1 < n && (c = cls(t[i + 1])) != 0) { j = i + 1; while (j < n && cls(t[j]) == c) j++; }
        else if ((c = cls(t[i])) != 0) { while (j < n && cls(t[j]) == c) j++; }
        else { while (j < n && cls(t[j]) == 0) j++; }
        k += encode_chunk(t + i, j - i, out + k);
        i = j;
    }
    return k;
}

// ---------------- sampling ----------------
static unsigned long long rng_state;
static float rand_f32(void) {
    rng_state ^= rng_state >> 12; rng_state ^= rng_state << 25; rng_state ^= rng_state >> 27;
    return ((rng_state * 0x2545F4914F6CDD1Dull) >> 40) / 16777216.0f;
}

static float *sort_keys;
static int cmp_desc(const void *a, const void *b) {
    float x = sort_keys[*(const int *)a], y = sort_keys[*(const int *)b];
    return (x < y) - (x > y);
}

static int sample(float *logits, float temp, int top_k, int *idx) {
    int V = cfg.vocab_size;
    if (temp <= 0) { int b = 0; for (int i = 1; i < V; i++) if (logits[i] > logits[b]) b = i; return b; }
    for (int i = 0; i < V; i++) { logits[i] /= temp; idx[i] = i; }
    sort_keys = logits;
    qsort(idx, V, sizeof(int), cmp_desc);
    int k = top_k < V ? top_k : V;
    float probs[k], m = logits[idx[0]], sum = 0;
    for (int i = 0; i < k; i++) { probs[i] = expf(logits[idx[i]] - m); sum += probs[i]; }
    float r = rand_f32() * sum;
    for (int i = 0; i < k; i++) { r -= probs[i]; if (r <= 0) return idx[i]; }
    return idx[k - 1];
}

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.bin tok.bin [-t temp] [-n tokens] [-s seed] [-i prompt] [-c chat]\n", argv[0]); return 1; }
    float temp = 0.8f; int steps = 256; unsigned long long seed = (unsigned long long)time(NULL);
    const char *prompt = NULL, *chat = NULL;
    for (int i = 3; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "-t")) temp = atof(argv[i + 1]);
        else if (!strcmp(argv[i], "-n")) steps = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "-s")) seed = strtoull(argv[i + 1], NULL, 10);
        else if (!strcmp(argv[i], "-i")) prompt = argv[i + 1];
        else if (!strcmp(argv[i], "-c")) chat = argv[i + 1];
    }
    rng_state = seed ? seed : 1;
    float *buf = load_model(argv[1]);
    load_tokenizer(argv[2]);

    if (n_tok > cfg.vocab_size) { fprintf(stderr, "tok.bin has %d tokens but the model only %d\n", n_tok, cfg.vocab_size); return 1; }
    size_t plen = (prompt ? strlen(prompt) : 0) + (chat ? strlen(chat) : 0);
    int *toks = malloc((plen + cfg.block_size + 3) * sizeof(int)), n = 0;   // a token is at least one byte
    if (chat) { toks[n++] = USER; n += encode(chat, toks + n); toks[n++] = ASSISTANT; }
    else if (prompt) { n = encode(prompt, toks); }
    if (n == 0) toks[n++] = EOT;                      // no prompt: "start of a new story"
    if (n >= cfg.block_size) { fprintf(stderr, "prompt too long\n"); return 1; }
    if (prompt) printf("%s", prompt);

    int *idx = malloc(cfg.vocab_size * sizeof(int)), generated = 0;
    double t0 = now();
    for (int pos = 0; pos < cfg.block_size; pos++) {    // stops where model.py's generate_cached stops
        float *logits = forward(toks[pos], pos);     // prefill runs through the prompt one token at a time
        if (pos < n - 1) continue;
        int next = sample(logits, temp, 50, idx);
        if (next == EOT || generated >= steps) break;
        toks[pos + 1] = next; n++; generated++;
        if (next < n_tok) { fwrite(vocab[next], 1, vlen[next], stdout); fflush(stdout); }
    }
    double dt = now() - t0;
    printf("\n");
    fprintf(stderr, "\n%d tokens in %.2fs (%.0f tok/s)\n", generated, dt, generated / dt);
    free(buf); free(toks); free(idx);
    return 0;
}
