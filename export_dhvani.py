"""
export_dhvani.py  --  DHVANI -> ESP32-S3 export
Usage:  python export_dhvani.py <n_trees> <output_folder>
Run inside the dhvani_app folder (needs dhvani_model_multiclass.pkl).

Writes into <output_folder>:
  dhvani_feat_tables.h  (mel filterbank, DCT matrix, librosa pad mode)
  dhvani_golden.h       (test frames' expected features + RF votes)
  dhvani_rf.h           (Random Forest, first n_trees trees)
"""
import sys, os
import numpy as np
import joblib
import librosa
import scipy.fft

SR, N, HOP = 16000, 1600, 512
NBIN, NMEL, NMFCC = N // 2 + 1, 128, 13
NTREES = int(sys.argv[1]) if len(sys.argv) > 1 else 25
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)


# ---- copied verbatim from dhvani_core.py (avoids importing torch) ----
def extract_features(chunk, sr=SR):
    n_fft = min(2048, len(chunk))
    mfcc = librosa.feature.mfcc(y=chunk.astype(np.float32), sr=sr, n_mfcc=13, n_fft=n_fft)
    n_frames = mfcc.shape[1]
    width = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
    width = max(width, 3)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta = librosa.feature.delta(mfcc, width=width)
    delta_mean = np.mean(delta, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std, delta_mean])
# -----------------------------------------------------------------------

melfb = librosa.filters.mel(sr=SR, n_fft=N, n_mels=NMEL).astype(np.float32)
dctm = scipy.fft.dct(np.eye(NMEL), type=2, norm="ortho", axis=0)[:NMFCC].astype(np.float32)


def ref_features(x, pad_reflect):
    """numpy twin of the C++ code (used to detect librosa's pad mode)."""
    xp = np.pad(x.astype(np.float32), N // 2, mode="reflect" if pad_reflect else "constant")
    nfr = 1 + (len(xp) - N) // HOP
    assert nfr == 4
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / N)).astype(np.float32)
    P = np.stack([np.abs(np.fft.rfft(xp[t * HOP:t * HOP + N] * win)) ** 2 for t in range(nfr)], 1)
    S = 10 * np.log10(np.maximum(1e-10, melfb @ P))
    S = np.maximum(S, S.max() - 80.0)
    mf = (dctm @ S).T  # (nfr, 13)
    mean = mf.mean(0)
    std = mf.std(0)
    dmean = (mf[2] + mf[3] - mf[0] - mf[1]) / 4.0
    return np.concatenate([mean, std, dmean])


def golden_frames():
    n = np.arange(N)
    a = 0.30 * np.sin(2 * np.pi * 440 * n / 16000) + 0.10 * np.sin(2 * np.pi * 1234.5 * n / 16000)
    st, b = 12345, []
    for _ in range(N):
        st = (st * 1103515245 + 12345) & 0xFFFFFFFF
        v = (st >> 16) & 0x7FFF
        b.append(0.2 * (v / 16384.0 - 1.0))
    b = np.array(b)
    c = 0.5 * a + 0.5 * b
    return [f.astype(np.float32) for f in (a, b, c)]


frames = golden_frames()
expected = [extract_features(f) for f in frames]

errs = {}
for mode in (True, False):
    errs[mode] = max(np.max(np.abs(ref_features(f, mode) - e)) for f, e in zip(frames, expected))
pad_reflect = errs[True] < errs[False]
best = errs[pad_reflect]
print("librosa", librosa.__version__, "| pad reflect err %.2e, constant err %.2e" % (errs[True], errs[False]))
if best > 5e-2:
    sys.exit("ERROR: numpy twin does not match librosa (err %.3g). Send this output to Claude." % best)
print("pad mode detected:", "reflect" if pad_reflect else "constant", "| twin error %.2e" % best)


def farr(name, vals):
    return "static const float %s[] = {%s};\n" % (name, ",".join("%.9g" % v for v in np.ravel(vals)))


# ---- tables header ----
starts, lens, offs, ws = [], [], [], []
for m in range(NMEL):
    nz = np.nonzero(melfb[m])[0]
    s, e = int(nz[0]), int(nz[-1])
    starts.append(s); lens.append(e - s + 1); offs.append(len(ws))
    ws.extend(melfb[m, s:e + 1].tolist())
with open(os.path.join(OUT, "dhvani_feat_tables.h"), "w") as f:
    f.write("#pragma once\n#include <stdint.h>\n")
    f.write("#define DHVANI_PAD_REFLECT %d\n" % (1 if pad_reflect else 0))
    f.write("static const uint16_t MEL_START[] = {%s};\n" % ",".join(map(str, starts)))
    f.write("static const uint16_t MEL_LEN[] = {%s};\n" % ",".join(map(str, lens)))
    f.write("static const uint16_t MEL_OFF[] = {%s};\n" % ",".join(map(str, offs)))
    f.write(farr("MEL_W", ws))
    f.write(farr("DCT_M", dctm))

# ---- Random Forest ----
m = joblib.load("dhvani_model_multiclass.pkl")
feat, thr, left, right, tstart = [], [], [], [], []
for e in m.estimators_[:NTREES]:
    t = e.tree_
    assert t.node_count < 65536
    tstart.append(len(feat))
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            feat.append(-1); thr.append(0.0)
            left.append(int(np.argmax(t.value[i][0]))); right.append(0)
        else:
            feat.append(int(t.feature[i])); thr.append(float(t.threshold[i]))
            left.append(int(t.children_left[i])); right.append(int(t.children_right[i]))


def carr(name, ctype, vals, fmt):
    return "static const %s %s[] = {%s};\n" % (ctype, name, ",".join(fmt % v for v in vals))


with open(os.path.join(OUT, "dhvani_rf.h"), "w") as f:
    f.write("#pragma once\n#include <stdint.h>\n")
    f.write("#define RF_TREES %d\n#define RF_FEATURES 39\n#define RF_CLASSES 4\n" % NTREES)
    f.write(carr("rf_start", "uint32_t", tstart, "%d"))
    f.write(carr("rf_feat", "int8_t", feat, "%d"))
    f.write(carr("rf_thr", "float", thr, "%.9g"))
    f.write(carr("rf_left", "uint16_t", left, "%d"))
    f.write(carr("rf_right", "uint16_t", right, "%d"))
    f.write("""
// votes[c] = number of trees voting for class c. Returns winning class.
static inline int rf_votes(const float* x, int* votes) {
  for (int c = 0; c < RF_CLASSES; c++) votes[c] = 0;
  for (int t = 0; t < RF_TREES; t++) {
    uint32_t b = rf_start[t]; uint32_t i = 0;
    while (rf_feat[b + i] >= 0)
      i = (x[rf_feat[b + i]] <= rf_thr[b + i]) ? rf_left[b + i] : rf_right[b + i];
    votes[rf_left[b + i]]++;
  }
  int best = 0;
  for (int c = 1; c < RF_CLASSES; c++) if (votes[c] > votes[best]) best = c;
  return best;
}
// p[c] = votes/RF_TREES (same role as sklearn predict_proba). Returns class.
static inline int rf_predict_proba(const float* x, float* p) {
  int v[RF_CLASSES];
  int best = rf_votes(x, v);
  for (int c = 0; c < RF_CLASSES; c++) p[c] = (float)v[c] / (float)RF_TREES;
  return best;
}
""")

# ---- golden header ----
gvotes = []
for e in expected:
    x32 = e.astype(np.float32).reshape(1, -1)
    v = [0, 0, 0, 0]
    for est in m.estimators_[:NTREES]:
        leaf = est.apply(x32)[0]
        v[int(np.argmax(est.tree_.value[leaf][0]))] += 1
    gvotes.append(v)
with open(os.path.join(OUT, "dhvani_golden.h"), "w") as f:
    f.write("#pragma once\n")
    f.write(farr("GOLD_FEAT", np.array(expected)))
    f.write("static const int GOLD_VOTES[3][4] = {%s};\n" % ",".join("{%s}" % ",".join(map(str, v)) for v in gvotes))

print("trees:", NTREES, "nodes:", len(feat), "approx KB:", len(feat) * 9 // 1024)
print("golden RF votes:", gvotes)
print("DONE ->", os.path.abspath(OUT))
