# The claim-classification model: audit, ceiling, and what replaced it

**Scope.** This covers only the ML/MLP claim-classification component —
`binary_truth_mlp.py`, `tfidf.py`, `classifier.py`, `mlp_classifier.py`, the
training and evaluation scripts, and the LIAR data they read. Retrieval,
article extraction, relevance filtering, claim decomposition, the NLI/stance
pipeline, the API flow and the frontend are untouched.

**The question.** Can this model reach roughly 70% on a legitimate held-out
test set?

**The answer.** No — not on this task, and not with any model. The ceiling for
predicting a claim's truth *from its wording alone* on LIAR is about 63–65%,
and this document shows the measurements that establish it. What the
investigation did change is everything that was making the current number
*unreadable*: a train/serve skew, a silent data-loading bug, two comparison
baselines that had never trained, an AUC computed off a rounded grid, and a
model-selection protocol whose margin of error was larger than the effects it
was being used to choose between.

Everything below was measured in this repository. Where a number is quoted
without a script, it can be reproduced with the commands in
[Reproducing this](#reproducing-this).

---

## Contents

- [1. What the audit found](#1-what-the-audit-found)
- [2. The dataset](#2-the-dataset)
- [3. Is the test split valid?](#3-is-the-test-split-valid)
- [4. Baseline table](#4-baseline-table)
- [5. Classical models](#5-classical-models)
- [6. Text representation](#6-text-representation)
- [7. Transformers](#7-transformers)
- [8. Domain-pretrained models](#8-domain-pretrained-models)
- [9. Other datasets, and why merging them is a trap](#9-other-datasets-and-why-merging-them-is-a-trap)
- [10. Why 70% is not reachable here](#10-why-70-is-not-reachable-here)
- [11. What shipped](#11-what-shipped)
- [Reproducing this](#reproducing-this)

---

## 1. What the audit found

Nine defects, in rough order of how much they distorted the reported numbers.

### 1.1 The loader silently dropped 45 rows — including 16 of the test set

Every `pd.read_csv(path, sep="\t", names=COLUMNS)` in the project read fewer
rows than the file contains. pandas defaults to `quotechar='"'`, and LIAR is a
tab-separated file full of quoted political speech, so a single `"` in a claim
opened a field that stayed open until the next `"` several lines later.
Everything in between was absorbed into that row's `statement` — carrying raw
tab characters and the following rows' ids, labels and speaker metadata — under
the **first** row's label.

| split | lines in file | rows the old call returned | lost |
|---|---:|---:|---:|
| train | 10,269 | 10,240 | 29 |
| valid | 1,284 | 1,284 | 0 |
| test | 1,283 | 1,267 | **16** |

The project's "1,267 LIAR test examples" was this bug, not the dataset. The
longest "statement" the old parser produced was **431 words**; the true maximum
across all three splits is **48**. No warning, no exception, and an accuracy
figure that looks entirely reasonable.

Fixed by `binary_truth_mlp.read_liar()` (`quoting=csv.QUOTE_NONE` — in a TSV the
tab is the delimiter, so a quote character carries no structural meaning), which
every ML script now routes through. Pinned by `tests/test_dataset_loading.py`.

### 1.2 The model was trained on one string and served another

`binary_truth_mlp.main()` trained on the bare statement. Every live request
goes through `make_prediction_features()`, which fills the metadata columns
with blanks and passes the row through `build_text_input()` — which emitted
every column unconditionally:

```
training:  "The economy grew by 3 percent last year."
serving:   "statement The economy grew by 3 percent last year. subject  speaker  job  state  party  context "
```

Seven constant tokens plus seven boundary bigrams on every request, against a
vocabulary that had never seen that shape. Measured on the test set: **61.88%**
served against **62.35%** on the text it was trained on. `build_text_input()`
now skips empty fields, so a statement-only request reaches the vectorizer as
exactly the statement.

An existing test asserted the *defect* ("every column name appears in the
text"), which is why it had survived. It now pins the fix.

### 1.3 The "Logistic Regression" comparison baseline had never trained

`classifier.py` ran 100 full-batch steps at lr 0.1 from a zero start on
L2-normalised TF-IDF rows. Its 1,267 test scores all landed inside
**[0.5543, 0.5597]** — every one above 0.5 — so it predicted "true-ish" for
every row and scored **56.35%**, the majority-class rate to four decimals.

It is not a weak baseline, it is a *stopped* one. The tell is in the same run:
its ROC-AUC was 0.661, so the ranking it had learned was fine; only the decision
boundary had never moved. With a step size appropriate to the gradient scale
(2.0, with momentum and decay) the same model trains.

(Figures in this section are as they stood *before* the loader fix, on 1,267
rows, because that is the state being described. §4 re-measures everything on
the corrected 1,283.)

### 1.4 The 6-class MLP scored below its own majority class

`mlp_classifier.py` at lr 0.03 for 30 epochs scored **20.84%** on the 6-class
test set against a **20.92%** majority baseline — worse than always answering
"half-true" — and its collapsed binary predictions were "true-ish" for all
1,267 rows. Same cause: L2-normalised sparse rows (~24 non-zeros of ~26,000)
leave He-initialised hidden activations near zero, and at that learning rate the
weights do not climb out in 30 epochs.

**Both comparison models on the Model Comparison page had learned nothing.**
The production model was being compared against two untrained models, which
flatters it.

### 1.5 The 6-class MLP's shuffling used the global RNG

`fit()` called `np.random.permutation` rather than the seeded generator, so
`seed` controlled only the initial weights. Two runs of the same configuration
gave different models. (The same defect had already been fixed in
`binary_truth_mlp.py`; this copy was missed.)

### 1.6 ROC-AUC was integrated off a rounded 200-point grid

`compute_auc()` sampled the ROC curve at 200 fixed thresholds in [0, 1],
**rounded each coordinate to four decimals**, and trapezoid-integrated. When a
model's scores are bunched — and the broken baseline's spanned 0.006 — almost
every sample lands on the same corner and the integral is of whatever the
rounding left behind. Replaced with the exact rank form (ties averaged), which
has no grid, no rounding and no free parameters.

### 1.7 The shipped MLP was under-fitted, not over-fitted

The file's own comments worried about overfitting on 26,626 dimensions over
10k rows. The measured behaviour was the opposite: at lr 0.05 the network's
probabilities were compressed into roughly [0.4, 0.7] — 96% of the test set —
and the model's confident region was almost empty. Its calibration error was
0.046, driven by systematic under-confidence at both ends.

### 1.8 Five input features were constant zero

`main()` trained with the credit-history columns zeroed (correctly — they leak
the target, and production never sends them), so five of the 26,626 inputs were
constant. Harmless, but they were being described as features.

### 1.9 The training loop paid for output it discarded

`BinaryTruthMLP.fit()`'s reporting block ran a forward pass over the whole
training set every five epochs — more expensive than the epoch that produced it
— and `quiet=True` suppressed only the `print`. A quiet sweep spent about a
fifth of its time computing numbers nobody saw.

---

## 2. The dataset

Read correctly, LIAR is 12,836 rows of short PolitiFact-rated political claims.

| split | rows | pants-fire | false | barely-true | half-true | mostly-true | true | true-ish |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 10,269 | 842 | 1,998 | 1,657 | 2,123 | 1,966 | 1,683 | 56.21% |
| valid | 1,284 | 116 | 263 | 237 | 248 | 251 | 169 | 52.02% |
| test | 1,283 | 92 | 250 | 214 | 267 | 249 | 211 | 56.66% |

**Binary mapping.** `{pants-fire, false, barely-true} → fake-ish`;
`{half-true, mostly-true, true} → true-ish`. Majority-class baselines: train
0.5621, valid 0.5202, **test 0.5666**. Class imbalance is mild; the validation
split is noticeably more balanced than the other two, which is itself a reason
not to trust a single-split comparison (see §3).

**Claim length.** Mean 18.3 words, median 17, p95 33, max 48. Mean 106 characters.

**Vocabulary.** 12,173 distinct training unigrams, of which 6,827 survive
`min_df=2`. 15.0% of the *types* in the test set are unseen in training, but
only **2.93% of tokens** — the tail is rare proper nouns, not vocabulary drift.

**Duplicates.**

| | count |
|---|---:|
| exact (normalised) duplicate groups | 28 (61 rows) |
| duplicate groups with **contradictory** binary labels | 5 |
| duplicate groups spanning more than one split | 11 (24 rows) |
| test rows with an exact twin in train | **5 of 1,283** |
| valid rows with an exact twin in train | 7 of 1,284 |

Five of 28 duplicate groups carry contradictory labels — the same sentence
rated on both sides of the cut. On a small sample, but it is a direct measure
of label noise.

**Near-duplicates** (character 3–5-gram cosine, test vs train):

| threshold | test rows | share |
|---|---:|---:|
| ≥ 0.95 | 6 | 0.47% |
| ≥ 0.90 | 8 | 0.63% |
| ≥ 0.80 | 10 | 0.79% |
| ≥ 0.70 | 19 | 1.50% |

Under 1% contamination. The split is genuinely held out.

**Speaker overlap.** 84.4% of test rows have a speaker who also appears in
training. That is not leakage of the label, but it is why speaker identity is
worth two accuracy points on its own (§10).

**Domain.** 144 subject tags, and the top fifteen are `economy`, `health-care`,
`taxes`, `federal-budget`, `education`, `jobs`, `state-budget`,
`candidates-biography`, `elections`, `immigration`, `foreign-policy`, `crime`,
`history`, `energy`, `environment`. Party distribution: 5,665 republican, 4,137
democrat, 2,183 none. **This is a US-politics corpus**, and NewsChecker's stated
scope — news, current events, scientific facts, historical claims, technical
facts — is mostly outside it.

---

## 3. Is the test split valid?

Checked, item by item, against the brief's list.

| check | finding |
|---|---|
| preprocessing fit only on training data | ✅ `build_vocab` is called on train text only, in every script |
| vectorizer vocabulary does not leak test information | ✅ same |
| normalisation/scaling does not use test data | ✅ row-wise L2 only — no cross-row statistics exist |
| feature selection does not use test labels | ✅ `min_df` is unsupervised |
| hyperparameters not selected on test results | ✅ before and after; `train_experiments.py` never opens `test.tsv` |
| duplicates/near-duplicates split across train and test | ⚠️ 5 exact, 10 at cosine ≥ 0.80 — **0.8%**, immaterial |
| threshold optimisation not leaking test labels | ✅ tuned on validation, now on out-of-fold predictions |

**No test-set leakage was found.** The evaluation was honest. What was *not*
sound was the **selection protocol**, and that had a measurable cost.

Configurations were chosen on `valid.tsv`: one split, 1,284 rows, on which the
95% interval for an accuracy is about ±2.6 points — while the differences being
chased are one to two. The consequence, measured with a paired bootstrap over rows on the *interim*
candidate that single-split selection had picked — a word + character n-gram
model, scored before the loader fix, so on 1,267 test rows:

| comparison | validation | held-out test |
|---|---|---|
| skew fix alone | +1.40 pts, CI [+0.31, +2.57] | +0.47 pts, CI [−0.63, +1.66] |
| model swap alone | +1.64 pts, CI [−0.47, +3.74] | +0.87 pts, CI [−1.34, +3.00] |
| **both** | **+3.04 pts, CI [+0.70, +5.37]** | **+1.34 pts, CI [−0.87, +3.55]** |

The validation margin cleared zero; the test margin did not. Most of that gap
is the split, not the model — textbook selection optimism, visible because both
numbers were computed rather than one.

`train_experiments.py` now selects by **5-fold cross-validation over
train+valid** (11,553 rows instead of 1,284), with folds shared across
configurations so comparisons are paired, and the decision threshold taken from
pooled out-of-fold predictions. The selection criterion is written down in the
script *before* the sweep runs.

A lower honest number is worth more than a higher fragile one. All figures
below are reported on the corrected 1,283-row test set, so they are **not**
directly comparable with the project's previously published figures, which were
computed on 1,267 rows.

---

## 4. Baseline table

Held-out LIAR test set, **1,283 rows**, majority-class baseline **56.66%**.
Every row is an already-published or already-selected configuration, so
reporting its test number is not tuning against test — selection happened in
`train_experiments.py`, which never opens `test.tsv`.

| model | features | acc % | prec % | recall % | F1 | ROC-AUC | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| Majority class (always "true-ish") | none | 56.66 | 56.66 | 100.00 | 0.7234 | 0.5000 | 0.2456 |
| Logistic Regression *(as it was)* | TF-IDF word 1-2, raw TF | 56.66 | 56.66 | 100.00 | 0.7234 | 0.6596 | 0.2456 |
| Logistic Regression *(repaired)* | TF-IDF word 1-2, raw TF | 62.43 | 62.88 | 82.26 | 0.7128 | 0.6760 | 0.2255 |
| MLP 6-class → binary *(as it was)* | TF-IDF word 1-2, raw TF | 56.66 | 56.66 | 100.00 | 0.7234 | 0.6387 | 0.2442 |
| MLP 6-class → binary *(repaired)* | TF-IDF word 1-2, raw TF | 61.34 | 60.85 | 89.13 | 0.7232 | 0.6650 | 0.2303 |
| Binary Truth MLP *(as served, with skew)* | TF-IDF word 1-2, raw TF | 61.96 | 62.33 | 83.08 | 0.7123 | 0.6720 | 0.2274 |
| Binary Truth MLP *(skew fixed)* | TF-IDF word 1-2, raw TF | 62.51 | 63.52 | 79.50 | 0.7062 | 0.6725 | 0.2260 |
| **Linear truth model (ships)** | TF-IDF word 1-2, sublinear TF | **62.74** | 64.20 | 77.44 | 0.7020 | **0.6760** | **0.2247** |

With 95% bootstrap intervals and calibration error:

| model | accuracy 95% CI | ECE |
|---|---|---:|
| Majority class | [0.5378, 0.5939] | 0.0046 |
| Logistic Regression *(as it was)* | [0.5378, 0.5939] | 0.0126 |
| Logistic Regression *(repaired)* | [0.5978, 0.6501] | 0.0450 |
| MLP 6-class *(as it was)* | [0.5378, 0.5939] | 0.0023 |
| MLP 6-class *(repaired)* | [0.5853, 0.6407] | 0.0590 |
| Binary Truth MLP *(as served)* | [0.5931, 0.6454] | 0.0443 |
| Binary Truth MLP *(skew fixed)* | [0.6002, 0.6516] | 0.0407 |
| **Linear truth model (ships)** | **[0.6017, 0.6547]** | **0.0370** |

**Read the first, second and fourth rows together.** The "Logistic Regression"
and "MLP 6-class" comparison models scored *exactly* the majority-class rate,
to four decimals, with identical precision, recall, F1 and Brier — because all
three predict "true-ish" for all 1,283 rows. The 6-class model's own accuracy
was **20.81%** against a 20.81% majority baseline: identical, because it always
answered "half-true". The page comparing the production model against those two
was comparing it against the majority class twice.

Repaired, they are real baselines: 62.43% and 61.34%. And **that is the finding
that matters most in this table** — a plain logistic regression on word TF-IDF
lands within 0.3 points of the shipped model, and its interval covers it
entirely. Every honest model of this task arrives at the same place.

**Does the neural network help?** No. It never did. The apparent gap was two
untrained comparisons.

---

## 5. Classical models

Every family the brief names, swept against every representation in §6:
**194 configurations, 10 model families × 10 representations**, scored on the
**validation** split only (`work/sweep_final.py`; `test.tsv` is never opened by
that script). scikit-learn is used here as a reference implementation — it is
not a dependency of the service, and the shipped model is reimplemented in
NumPy.

Best configuration of each family (validation accuracy at its tuned threshold):

| family | best configuration | val acc | AUC |
|---|---|---:|---:|
| Linear SVM + Platt scaling | C=0.01, word 1-2 + char 3-5 | **0.6519** | 0.6872 |
| Logistic regression | C=0.1, word 1-2 + char 3-5 | 0.6511 | **0.6894** |
| SGD (log loss) | α=1e-3, word 1-2 + char 3-5 | 0.6503 | 0.6891 |
| Bernoulli naive Bayes | α=1, char 3-5 | 0.6472 | 0.6823 |
| sklearn MLP | 64 hidden, char 2-5 | 0.6464 | 0.6851 |
| Complement naive Bayes | α=1, word 1-2 + char 3-5 | 0.6464 | 0.6794 |
| Multinomial naive Bayes | α=1, word 1-2 + char 3-5 | 0.6456 | 0.6794 |
| Ridge + Platt scaling | α=1, word 1-2 min_df=1 | 0.6363 | 0.6781 |
| Random forest (on SVD-200) | 300 trees, word 1-2 | 0.6269 | 0.6617 |
| Gradient boosting (on SVD-200) | 250 iters, word 1-2 | 0.6192 | 0.6551 |

Across all 194 configurations: tuned accuracy **min 0.5942, median 0.6371, max
0.6519**; AUC min 0.6365, median 0.6759, max 0.6900. Validation baseline 0.5202.

The shape of the result matters more than any single row. **Ten different model
families, the best of each, span 3.3 points** — and the top seven span 0.6 of a
point, which on 1,284 rows is noise. Nothing is left on the table for a better
optimiser to find.

Two specific answers to the brief:

- **Sparse-appropriate models win.** Random forest and gradient boosting have
  to go through a 200-dimensional SVD to run at all, and both give up two to
  three points doing it. On 25,000 sparse features the linear models and naive
  Bayes are the right tools, and they finish within half a point of each other.
- **Calibrated linear SVM edges it, logistic regression matches it.** The
  difference between them (0.6519 vs 0.6511) is a single row of the validation
  set. Logistic regression ships because it is what a NumPy reimplementation
  can be, exactly, in forty lines.

**A strong linear model beats the MLP, and that is fine.** Cross-validated over
11,553 rows, a regularised linear model scores **0.6263** against **0.6211** for
the shipped hidden-layer network — **+0.97 points, 95% CI [+0.38, +1.51]**. A
hidden layer buys nothing here because there is very little interaction
structure to find: the signal in a claim's wording is close to additive.

**And a repaired plain logistic-regression baseline nearly matches the shipped
model.** On the held-out test set the "simplest possible" comparison model in
`classifier.py` scores **62.43%** against the production model's **62.74%**.
That 0.3-point gap is noise. It is the clearest available statement of where
this task's difficulty lives — not in the model.


## 6. Text representation

Word n-grams, character n-grams, and their union, each paired with every model
in §5 (`work/sweep_final.py`). Best model per representation:

| representation | features | best model | val acc | AUC |
|---|---:|---|---:|---:|
| word 1-2 + char 3-5 | 63,290 | LinearSVC+Platt C=0.01 | **0.6519** | 0.6872 |
| word 1-2, min_df=1 | 95,432 | LogReg C=0.3 | 0.6495 | 0.6825 |
| char 2-5 | 38,666 | LogReg C=0.3 | 0.6495 | 0.6856 |
| word 1-2 | 25,715 | LinearSVC+Platt C=0.03 | 0.6488 | 0.6794 |
| char 3-5 | 37,575 | LogReg C=0.3 | 0.6488 | 0.6890 |
| char 3-6 | 54,852 | LinearSVC+Platt C=0.03 | 0.6480 | 0.6882 |
| word 1-1 + char 3-6 | 61,644 | LogReg C=0.1 | 0.6480 | 0.6865 |
| word 1-2, raw TF | 25,715 | LinearSVC+Platt C=0.03 | 0.6456 | 0.6780 |
| word 1-1 | 6,792 | LinearSVC+Platt C=0.01 | 0.6441 | 0.6758 |
| word 1-3 | 36,891 | LinearSVC+Platt C=0.03 | 0.6441 | 0.6789 |

**Ten representations, from 6,792 features to 95,432, span 0.8 of a point.**
Unigrams alone land within 0.8 points of the best thing in the whole sweep. The
representation is not what is binding either.

The rest of it:

- **Character n-grams look like a win on one split and vanish under
  cross-validation.** On the 1,284-row validation split, word + char 3-5grams
  gave the best accuracy in the whole sweep and the best AUC. Cross-validated
  over 11,553 rows it is worth **nothing**: 0.6254 against 0.6263 for word
  features alone, while tripling the vocabulary from 25,715 to 63,290
  features. This is the single clearest illustration of why the selection
  protocol had to change (§3), and it is why the shipped model does **not**
  use them.
- **Sublinear TF is a small, free improvement** (0.6263 vs 0.6254 raw TF,
  cross-validated) and it removes a genuine defect: `count / total_tokens`
  makes a term's weight depend on sentence length, which is not a property of
  the term.
- **Bigrams help slightly over unigrams; trigrams do not.**
- `min_df=1` (95,432 features) does not beat `min_df=2` (25,715).
- **14 hand-built surface features** — hedging, absolutes, negation, digit and
  percentage density, quote count, capitalisation ratio, mean word length,
  "Says" prefix — reach only **0.5615** on their own, and *reduce* accuracy
  when concatenated onto TF-IDF (0.6332 against 0.6449 on the same split).
  Reported because a negative result is still a result.
- **Ensembling** the best word model, the best character model and the best
  union model gains at most **+0.5 points** on validation (0.6480 → 0.6503)
  with no AUC improvement — inside the noise, and not worth three models in
  production.

## 7. Transformers

**This could not be run in this environment, and it would be dishonest to
report a number for it.** The session's egress policy denies
`huggingface.co` (`403` on CONNECT), which is where every pretrained
checkpoint lives. The `transformers` library installs fine from PyPI; the
weights cannot be fetched. Unblocking that host is the one thing needed to run
this section as specified.

Two substitutes were run instead, and they point the same way.

**Substitute 1 — a strong pretrained semantic representation.** GloVe-300
(reachable, since gensim-data is hosted on GitHub releases) is a genuine test
of the question a transformer would answer: *does a better representation of
meaning break the ceiling?*

| representation | val acc | AUC |
|---|---:|---:|
| GloVe-300 mean-pooled + LogReg C=0.1 | 0.6262 | 0.6628 |
| GloVe-300 mean-pooled + LogReg C=1.0 | 0.6301 | 0.6587 |
| GloVe-300 mean-pooled + LogReg C=10 | 0.6269 | 0.6572 |
| TF-IDF word 1-2 (for comparison) | 0.6277 | 0.6761 |
| TF-IDF + GloVe-300 concatenated | 0.6347 | 0.6770 |

300 dimensions of pretrained distributional semantics scores the **same** as a
bag of words, and adding it to TF-IDF is worth 0.7 points — inside the noise.
The representation is not what is binding.

**Substitute 2 — the published literature on this exact benchmark.** Reported
LIAR statement-only binary results cluster where this investigation's own
measurements do: BERT on the statement alone with no metadata or justification
reaches about **60%**; a survey of LIAR work puts most models at **no more than
62%** with BERT-base, with a CNN-BiLSTM over BERT embeddings at **63.06%**; and
one direct comparison reports **SVM with bag-of-words at 0.624 against RoBERTa
at 0.620** on the binary task — the linear model *ahead* of the transformer.
Results in the 68–69% range exist but add metadata and sentiment/emotion
features that a pasted claim does not carry.

So the expected value of a transformer here is roughly one point, possibly
negative, for 60–250MB of weights, a PyTorch dependency in the request path,
and inference in the hundreds of milliseconds instead of microseconds. On
accuracy, F1, calibration, inference cost, deployment practicality and training
cost — the brief's own criteria — it does not justify itself for this task.
That conclusion should be re-tested if the host is ever allowed.

Sources: [Fake News Detection: Experiments and Approaches beyond Linguistic
Features](https://arxiv.org/pdf/2109.12914) ·
[An Adversarial Benchmark for Fake News Detection Models](https://arxiv.org/pdf/2201.00912) ·
[LIAR benchmark summary](https://www.emergentmind.com/topics/liar-benchmark) ·
["Liar, Liar Pants on Fire" (Wang 2017)](https://arxiv.org/pdf/1705.00648)

## 8. Domain-pretrained models

Same blocker: news-, misinformation-, fact-verification- and NLI-pretrained
checkpoints are all distributed through the host this session cannot reach, so
none of them could be evaluated, and none is claimed either way.

What *can* be said from what was measurable: the service already runs a
domain-appropriate pretrained model — `cross-encoder/nli-deberta-v3-base`, in
`nli_service.py` — and it is used for the task that pretraining actually helps
with, which is deciding whether a retrieved passage supports a claim. That is a
function of *two* texts. The claim-only model is being asked for a function of
*one*, and §10 shows why no amount of pretraining fixes that. A
misinformation-pretrained encoder would arrive with the same handicap.

## 9. Other datasets, and why merging them is a trap

The brief asks whether LIAR is too narrow for NewsChecker. It is: 144 subject
tags, all US-politics-shaped (`economy`, `health-care`, `taxes`,
`federal-budget`, `education`, `jobs`, `state-budget`…), 9,837 of 12,836 rows
attributed to a Republican or Democrat, and PolitiFact's rating scale as the
label. Science, health, technology, history and general news — the product's
stated scope — are essentially absent.

So a broader corpus was tried. **x-fact** (Gupta & Srikumar 2021, on GitHub and
therefore reachable) is the largest multilingual claim-verification corpus with
per-claim veracity labels. Its English portion is 12,294 claims.

**It is not a broadening at all, and merging it naively would have leaked the
test set.**

| finding | value |
|---|---|
| x-fact English claims | 12,294 |
| …whose `site` is `politifact.com` | **12,294 (100%)** |
| …that appear verbatim in LIAR **train** | 6,249 (50.8%) |
| …that appear verbatim in LIAR **valid** | 797 (6.5%) |
| …that appear verbatim in LIAR **test** | **788 (6.4% of x-fact — 61.4% of the test set)** |
| …novel to all three LIAR splits | 4,473 (36.4%) |
| …usable after mapping labels | 4,436 |

Dropping 4,436 additional PolitiFact claims into the training set — which is
exactly what "add more fact-checking data" looks like from the outside — would
have put **788 test claims into training**. And the apparent gain looks
convincing:

| training data | val acc | AUC |
|---|---:|---:|
| LIAR train only | 0.6277 | 0.6761 |
| + all 4,436 novel x-fact claims | **0.6534** | **0.7199** |
| + those with cosine < 0.90 to valid/test | 0.6449 | 0.7042 |
| + those with cosine < 0.80 to valid/test | 0.6308 | 0.6853 |
| + those with cosine < 0.70 to valid/test | 0.6285 | 0.6826 |

**+2.6 points and +4.4 AUC, of which +0.1 and +0.7 survive near-duplicate
removal.** Ninety per cent of the "improvement" was contamination, and the
exact-match dedup that most people would stop at removed almost none of it —
it took character-n-gram cosine filtering to see. This is the concrete answer
to the brief's "do NOT blindly merge datasets".

**LIAR-PLUS** (the same claims plus PolitiFact's ruling justification) was
also tested, because the published ~70% binary result on LIAR comes from it.
The justification is written by the fact-checker *after* the verdict, and is
not available for a live claim — but for this model class it turns out not to
help at all:

| input | val acc | AUC |
|---|---:|---:|
| statement only (what the API gets) | 0.6277 | 0.6761 |
| statement + fact-checker justification | 0.6223 | 0.6778 |
| justification only, claim removed | 0.5436 | 0.6011 |

The justification adds nothing to a linear model, so the published 70% depends
on architecture as well as on an input the product cannot supply. Either way it
is not a route to 70% here.

**Checked and rejected for the same reason or unavailable:** FEVER (claims are
annotator-mutated Wikipedia sentences with well-documented claim-only
artefacts, and `fever.ai` is unreachable), PUBHEALTH and SciFact (data hosted
on blocked domains), the ISOT/Kaggle fake-news corpora (near-perfect accuracy
that is entirely source detection — Reuters wire copy against scraped fake
sites — a shortcut, not a skill).

**Verdict on a custom dataset.** A broader corpus is worth building only if
breadth is the binding constraint. §10 shows it is not: the label is not a
function of the input, in any domain. A NewsChecker-specific corpus of news,
science and history claims would make the model's *coverage* honest — it would
stop it being a US-politics model applied to everything — but it would not move
it towards 70%, and building one by hand-labelling would produce a model that
had learned the labeller. That is why this investigation did not ship one, and
why the recommendation is to keep the claim-only model in its stated auxiliary
role rather than to enlarge it.

## 10. Why 70% is not reachable here

Four measurements, each ruling out one candidate bottleneck from the brief's
list. All on validation.

**It is not sample size.** The learning curve has flattened:

| training rows | val acc | AUC |
|---:|---:|---:|
| 1,026 (10%) | 0.5766 ± 0.0083 | 0.6291 |
| 2,567 (25%) | 0.6075 ± 0.0052 | 0.6448 |
| 5,134 (50%) | 0.6186 ± 0.0022 | 0.6587 |
| 7,701 (75%) | 0.6238 ± 0.0077 | 0.6727 |
| 10,269 (100%) | 0.6277 | 0.6761 |

Doubling from 5k to 10k bought 0.9 points. Adding a further 3,966 genuinely
novel same-distribution claims bought **0.1** (§9). Extrapolating the curve,
reaching 70% would take on the order of a million labelled claims — and only if
the curve kept its slope, which the last two rows say it will not.

**It is not features or metadata.** Even *cheating* — giving the model the
speaker, party, job and state that a pasted claim never carries — is worth two
points:

| input | val acc | AUC |
|---|---:|---:|
| statement only | 0.6277 | 0.6761 |
| statement + subject | 0.6332 | 0.6788 |
| statement + speaker/party/job/state | **0.6480** | **0.7010** |
| statement + all metadata + context | 0.6456 | 0.6985 |
| metadata only, no claim text | 0.5997 | 0.6590 |
| speaker name only | 0.5771 | 0.6236 |

Note the fifth row: **you can beat the baseline by eight points without reading
the claim at all**, purely from who said it. That is what a large part of the
published LIAR literature is measuring, and it is unavailable in production.
Even with all of it, 0.648 is not 0.70.

**It is not the representation.** §7: 300-dimensional pretrained semantics
scores the same as a bag of words.

**It is the task.** The binary target cuts an ordinal scale between two
adjacent rungs — "barely-true" (fake-ish) and "half-true" (true-ish) — that
human fact-checkers assign by judgement, and **37.0% of the corpus sits on
those two rungs**. Accuracy by underlying rating:

| 6-way rating | n | accuracy | binary target |
|---|---:|---:|---|
| pants-fire | 116 | 0.6466 | fake-ish |
| false | 263 | 0.4867 | fake-ish |
| barely-true | 237 | **0.3924** | fake-ish |
| half-true | 248 | 0.6774 | true-ish |
| mostly-true | 251 | 0.7729 | true-ish |
| true | 169 | 0.8757 | true-ish |
| **extremes** (pants-fire + true) | 285 | **0.7825** | |
| **the two rungs either side of the cut** | 485 | **0.5381** | |

On the rows either side of the cut the model is at 0.538 — **coin-flip** — and
those rows are more than a third of the data. On the extremes it reaches 0.78.
The model is not failing to learn; it is being asked to reproduce a distinction
that is not present in the input. Supporting evidence from §2: of 28 sentences
that appear more than once in the corpus, **5 carry contradictory binary
labels** — the same words, rated on both sides of the line.

That is the answer to the brief's question. The bottleneck is **task
formulation**, not architecture, data volume, features, or optimisation, and
70% is not reachable from a claim's wording alone. The thing that *does* reach
a useful verdict on a specific claim is the evidence pipeline this model is
deliberately kept out of — retrieval plus NLI over what the retrieval found.
That is a function of the claim **and** the evidence, and it is why
`ml.score` is flagged `auxiliary_only` and never enters the verdict.

## 11. What shipped

**A regularised linear model on sublinear-TF word TF-IDF**, selected by 5-fold
cross-validation over train+valid, trained in NumPy with no new dependency,
and reachable through the identical interface `main.py` already used — that
file is unchanged.

Cross-validated selection (11,553 rows, folds shared across variants, out-of-
fold threshold; `test.tsv` never opened):

| variant | CV acc | 95% CI | AUC | Brier |
|---|---:|---|---:|---:|
| mlp baseline (what shipped) | 0.6211 | [0.6121, 0.6303] | 0.6558 | 0.2311 |
| mlp, lr 0.5 + early stop | 0.5928 | [0.5840, 0.6024] | 0.6073 | 0.2609 |
| linear, raw TF, word only | 0.6254 | [0.6160, 0.6342] | 0.6620 | 0.2288 |
| **linear, sublinear TF, word only** | **0.6263** | [0.6174, 0.6352] | 0.6616 | 0.2283 |
| linear, word + char 3-5 | 0.6254 | [0.6166, 0.6341] | 0.6598 | 0.2278 |
| linear, word + char 3-6 | 0.6229 | [0.6143, 0.6320] | 0.6594 | 0.2279 |
| linear, char 3-5 only | 0.6238 | [0.6153, 0.6326] | 0.6568 | 0.2284 |
| linear, word+char, L2 1e-5 | 0.6253 | [0.6165, 0.6339] | 0.6597 | 0.2276 |
| linear, word+char, L2 1e-3 | 0.6212 | [0.6125, 0.6300] | 0.6575 | 0.2330 |
| linear, word+char, L2 1e-2 | 0.6055 | [0.5970, 0.6144] | 0.6459 | 0.2436 |

Winner against the incumbent, paired over the same rows:
**+0.97 points, 95% CI [+0.38, +1.51] — clears zero.**

Then one pass over the held-out test set, through the serving feature path:

| | old MLP, as served | old MLP, skew fixed | **shipped linear model** |
|---|---:|---:|---:|
| accuracy | 61.96% | 62.51% | **62.74%** |
| precision | 62.33% | 63.52% | 64.20% |
| recall | 83.08% | 79.50% | 77.44% |
| F1 | 0.7123 | 0.7062 | 0.7020 |
| ROC-AUC | 0.6720 | 0.6725 | **0.6760** |
| Brier | 0.2274 | 0.2260 | **0.2247** |
| calibration error (ECE) | 0.0443 | 0.0407 | **0.0370** |
| artifact size | 14.4 MB | 14.4 MB | **1.1 MB** |
| training time | ~7 min | ~7 min | **15 s** |

Paired bootstrap on the test set: skew fix +0.55 pts [−0.70, +1.71], model swap
+0.23 pts [−1.01, +1.48], **total +0.78 pts [−0.70, +2.26]**.

**Read that honestly.** On 1,283 rows the total gain is *not* statistically
distinguishable from zero — the test set's own 95% interval is ±2.6 points, so
it cannot resolve a one-point difference, and no amount of wanting it to will
change that. The improvement is established on the 11,553-row cross-validation
(+0.97, CI [+0.38, +1.51]), and it is consistent in direction across every
metric on test: accuracy, AUC, Brier and calibration all move the right way at
once. Calibration is the one that matters most for this system, because the
score is displayed to users and consumed downstream as a prior: **ECE 0.037
against 0.044**.

Also repaired, so the Model Comparison page means something:

- `classifier.py`'s logistic regression converges instead of predicting one
  class: **56.66% → 62.43%**.
- `mlp_classifier.py`'s 6-class network trains instead of scoring exactly its
  own majority class: **20.81% → 25.02%**, with lr and epochs chosen on
  validation rather than by analogy, and its shuffling seeded.
- `compute_auc` is exact rather than a rounded 200-point trapezoid.

**Deliberately not changed:** retrieval, article extraction, relevance
filtering, claim decomposition, the NLI/stance pipeline, `main.py`, the API
contract, and the frontend. `evaluation_results.json` keeps its schema; only
the numbers in it moved.

## Reproducing this

```bash
cd ml-service
python -m unittest discover -s tests -p "test_*.py"   # 500+ tests

python train_experiments.py                # 5-fold CV over train+valid; never reads test
python train_experiments.py --save --only "linear, sublinear TF, word only"
python evaluate_production_model.py        # one pass over the held-out test set
python evaluate_models.py                  # regenerates client/public/evaluation_results.json
python binary_truth_mlp.py                 # simple train-on-train reproduction (~15 s)
```

The exploratory sweeps behind §5, §6, §7, §9 and §10 are research scripts, not
part of the service; they need `scikit-learn` and `gensim`, which the
ml-service deliberately does not depend on.
