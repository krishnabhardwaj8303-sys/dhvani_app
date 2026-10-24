# meter_rtl v2 (Sky130 tamper-detect meter core)

Run everything: `./run_regression.sh`  (add `--no-train` to reuse the .vh / vec_*.hex files)

## Status (as of this snapshot)
| Item | Status |
|---|---|
| Regression (feat, nn, top, bus) | PASS, 231 windows end-to-end, 0 mismatches |
| Integer-chain accuracy | 86.42% (regenerated weights; original weights.vh was not in the zip, original was ~87.0%) |
| APB wrapper meter_apb.v | PASS (tb_bus.v) |
| Area (yosys generic gates, NOT sky130) | 12,412 -> 10,410 cells (-16.1%), bit-exact. Window latency +~500 clocks |
| Formal NN (formal/nn_prove.sby, task prove) | PASS: done is 1 clock, start ignored while busy, no accumulator overflow (weight-based bound), k-induction |
| Formal feat (formal/feat_prove.sby, tasks abs_prove + mult_lemma) | PASS: f_valid 1 clock, dropped sample ignored, accumulator bounds. Uses an assume-guarantee abstraction of the 12x12 ACC multiplier; the bound is proven by task mult_lemma |

Run formal: `cd formal && sby -f nn_prove.sby prove && sby -f feat_prove.sby mult_lemma && sby -f feat_prove.sby abs_prove` (needs sby + z3; abs_prove takes ~2 min).

## Not done yet
Model improvement (longer window, new features, more tamper classes), ADC effects, SDC / DFT / reset synchroniser, PUF-bound weights, datasheet / integration guide, OpenLane Sky130 run.

## APB register map (word addressed)
0x00 CTRL [0]EN [1]SOFT_RST [2]IRQ_EN | 0x04 STATUS [0]READY [1]OVERRUN [2]DONE(w1c) [3]ALERT
0x08 SAMPLE [11:0]V [27:16]I | 0x0C RESULT [1:0]CLS [31:16]WINDOWS | 0x10 VERSION | 0x14 DROPPED
