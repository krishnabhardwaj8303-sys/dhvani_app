import joblib, numpy as np, sys
N = int(sys.argv[1]) if len(sys.argv) > 1 else 15
m = joblib.load('dhvani_model_multiclass.pkl')
feat, thr, left, right, starts = [], [], [], [], []
for e in m.estimators_[:N]:
    t = e.tree_
    assert t.node_count < 65536
    starts.append(len(feat))
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            feat.append(-1); thr.append(0.0)
            left.append(int(np.argmax(t.value[i][0]))); right.append(0)
        else:
            feat.append(int(t.feature[i])); thr.append(float(t.threshold[i]))
            left.append(int(t.children_left[i])); right.append(int(t.children_right[i]))
def arr(name, ctype, vals, fmt):
    return "const %s %s[] = {%s};\n" % (ctype, name, ",".join(fmt % v for v in vals))
with open('dhvani_rf.h', 'w') as f:
    f.write("#pragma once\n#include <stdint.h>\n")
    f.write("#define RF_TREES %d\n#define RF_FEATURES 39\n#define RF_CLASSES 4\n" % N)
    f.write(arr("rf_start", "uint32_t", starts, "%d"))
    f.write(arr("rf_feat", "int8_t", feat, "%d"))
    f.write(arr("rf_thr", "float", thr, "%.7g"))
    f.write(arr("rf_left", "uint16_t", left, "%d"))
    f.write(arr("rf_right", "uint16_t", right, "%d"))
    f.write("""
static inline int rf_predict(const float* x) {
  int votes[RF_CLASSES] = {0};
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
""")
print("trees:", N, "nodes:", len(feat), "approx KB:", len(feat) * 9 // 1024)
