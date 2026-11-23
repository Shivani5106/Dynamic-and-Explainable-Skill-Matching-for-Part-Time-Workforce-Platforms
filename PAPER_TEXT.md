# Paste-ready text for the paper

All numbers are traceable to `results/PAPER_RESULTS.md` (Tables A-G) and were produced by `run_experiments.py`.
Every result comes from a synthetic simulation, and the text below says so on purpose. If you remove that wording, a
reviewer who reads the code will notice.

---

## 1. Replace the abstract's last four sentences

> We implemented the framework as a working prototype and evaluated it in a controlled, time-aware simulation
> (600 workers, 600 evaluation tasks per run, six epochs, 20 random seeds). The prototype comprises skill extraction against a
> taxonomy, claimed/verified/demonstrated skill status, a weighted compatibility score with a feasibility gate,
> an evidence-based proficiency update, factor-level explanations, stable two-sided assignment and a web dashboard.
> Against keyword, semantic, supervised-ranking and two-sided-preference baselines, the framework reached NDCG@5 of 0.903
> versus 0.261-0.792, and 0.847 for a weight-tuned claims-only control. Ablations attribute +0.020 NDCG@5 to the
> verification layer and +0.006 to profile updates. Verification also reduced the representation of workers with less access to
> verification by 3.9 percentage points, a fairness cost that we quantify. Results are simulation-based and require
> validation on real platform data.

Also delete from the abstract: "Since the proposed framework has not yet been implemented, no experimental results are claimed."

---

## 2. Section IV (replace the current opening and IV.A-IV.B)

### IV. Implementation and Evaluation

**A. Prototype implementation.** The framework is implemented in Python (NumPy, scikit-learn, Flask). The skill layer
uses a taxonomy of 48 skills in 8 clusters with a skill-similarity graph (identical skill 1.0, same cluster 0.5,
selected cross-cluster links 0.1-0.3). Each worker-skill pair carries a status (none, claimed, verified, demonstrated, rejected).
The compatibility score M(w,t) follows Section III-A, preceded by a feasibility gate on availability, location and mandatory
verified evidence. The profile update p_new(s) = (1-lambda_eff)p_old(s) + lambda_eff E(s) is applied with lambda = 0.3 and a feedback
reliability of 0.8 (so lambda_eff = 0.24); the first piece of evidence for a skill initialises p(s). Explanations list each factor's
contribution w_i S_i, flag blocking factors, and distinguish verified from merely claimed required skills. Final assignment uses
task-proposing deferred acceptance with task-side scores M(w,t) and worker-side preference scores. A dashboard (Fig. 5) shows the top-5 workers per task with
explanations and accepts ratings that update profiles immediately.

**B. Experimental setup.** Real data containing verified proficiency and task outcomes are not publicly available, so we evaluate in a
generative simulation in which ground truth is known. Each of 600 workers has a hidden true proficiency per skill, a set of claimed skills,
availability over 21 weekly slots, a location zone, category and pay preferences, and years of experience. 35% of workers inflate their claims with 2-4 skills they do not hold.
Workers can submit evidence for claimed skills (probability 0.80 for group A, 0.50 for group B). The simulated verifier accepts a true skill with
probability 0.92 and a false claim with probability 0.08. A worker-task pair is *suitable* if the worker's true mean proficiency on the required skills meets the
task difficulty, the worker is available and within travel range, and the task satisfies the worker's category and pay preferences. NDCG uses graded true suitability.
The platform never observes true proficiency; it sees claims, verification evidence, stated preferences and noisy outcomes (noise SD 0.12).

Evaluation is time-aware: in each of six epochs, 100 new tasks are ranked using only profiles built from earlier epochs. After each epoch, 300 historical hires made by claim-based
matching (a static platform) are completed and their noisy outcomes are fed back. Weights were tuned by random search (600 candidate vectors per variant) on five separate development seeds and
evaluated on 20 test seeds; the sensitivity sweeps use a further 10 seeds. Baselines were (i) TF-IDF cosine on skill text, (ii) skill-graph semantic similarity (standing in for
embedding-based matching), (iii) a gradient-boosted ranker trained on earlier epochs' claim-based features and outcomes, with availability and preference factors applied,
(iv) a two-sided preference baseline, and (v) a control with the same weight-tuning budget but only claim-based terms. We report P@1, P@5, R@5 (capped at the number of suitable workers),
F1@5 and NDCG@5 with 95% confidence intervals over seeds, and paired tests for the ablations.

**C. Results.**

*Matching quality (Table A).* The full framework reached NDCG@5 = 0.903 ± 0.002 and P@1 = 0.903 ± 0.010. The keyword and semantic baselines reached 0.268 and 0.261, because they ignore
availability and preferences, which define suitability. The two-sided preference baseline reached 0.792 and the supervised ranker 0.752 ± 0.042. The most informative comparison is the weight-tuned claims-only control, at 0.847: the
full framework improves on it by 0.056 (RQ1).

*Ablations (Table B).* Removing verification lowered NDCG@5 by 0.020, removing profile updates by 0.006, and removing the feasibility gate by 0.005; all
differences were in the same direction in 20/20 seeds. Removing bilateral preferences lowered NDCG@5 by 0.131, but this largely reflects our simulator defining suitability to require
preference compatibility, so it should not be read as independent evidence for that component.

*Profile adaptation (Table C, RQ2).* With updates, NDCG@5 rose from 0.889 at cold start to 0.908 at epoch 5. Without updates it stayed between 0.894 and 0.899. The adaptation gain at epoch 5 was therefore about 0.013,
which is small. Verification helped at cold start (0.889 against 0.851 without it), when no outcome history exists. The two mechanisms overlap: either alone gives most of the improvement over the claims-only control, and
each adds less on top of the other.

*Robustness (Table D).* The gain from verification grew with the share of workers who inflate claims, from +0.013 (no inflation) to +0.027 (60% inflation). The claims-only control fell from 0.871 to 0.832 as inflation rose, while the full framework moved only from 0.906 to 0.899.
NDCG@5 varied by at most 0.005 across lambda in {0.1, 0.3, 0.5, 0.9}, and ±30% random perturbations of the weights left it unchanged at 0.903. The tuned weights were dominated by preference (0.82), with
skill 0.09, availability 0.06, verified capability 0.02 and performance 0.01, i.e. a near-lexicographic ordering in which the smaller terms break ties within preference-compatible workers; the weights should not be read as importance scores.

*Assignment (Table G).* Under deferred acceptance, 85.4% ± 1.1% of assigned pairs were suitable, against 66.9% for the two-sided preference baseline and 59.9% for the supervised ranker.

*Fairness (Tables E-F, RQ3 part 2).* Because group B was given lower verification access, the full framework under-represented group B in the top-5 by 3.9 percentage points
(31.3% of slots against 35.2% of suitable workers), whereas the variant without verification showed no gap (+0.9 points). The gap grew to 8.0 points when group B's verification access fell to 0.20
and vanished at equal access. Precision@5 was 0.758 for group A slots and 0.743 for group B slots. Verification therefore improves accuracy but can disadvantage workers who cannot readily obtain evidence;
alternative evidence routes (for example, short skill assessments) and routine fairness audits are needed.

*Efficiency and explanation.* Producing a ranked recommendation with factor-level explanations took 15 ms per task on one CPU core with 600 workers, and a profile update 0.005 ms.
Explanation coverage is 100% by construction. Because M(w,t) is additive, factor contributions are exact, so we do not report explanation faithfulness as an empirical result. User trust was not measured; it requires a user study.

**D. Threats to validity (replace the current list or add to it).**
(1) All results come from a simulator whose assumptions we chose (inflation rate, verifier accuracy, definition of suitability); different assumptions would change the numbers.
(2) Verification-quality statistics reflect the assumed verifier accuracy and are consistency checks, not findings.
(3) Feedback is attributed equally to every required skill of a task, which is a simplification.
(4) The supervised baseline learns from logs of claim-based hires, which are a selected sample.
(5) The embedding baseline uses a skill graph rather than a pretrained sentence encoder.
(6) Confidence intervals and p-values describe variation across simulation seeds, not real-world uncertainty.
(7) Group B's lower verification access is an assumption used to probe fairness, not a measured fact.

---

## 3. Replace the first Section V sentences

> This paper reviewed AI/ML/NLP-based approaches to worker-job matching, identified an integration gap, and presented an implemented prototype of a framework that separates claimed
> from verified skills, updates profiles from demonstrated outcomes, incorporates two-sided preferences and produces factor-level explanations. In a controlled simulation, the
> framework improved NDCG@5 over a tuned claims-only control by 0.056, with verification contributing most (+0.020) and profile updates a smaller, growing gain (+0.006 on average, about +0.013 by the fifth epoch).
> Verification also reduced the representation of workers with limited access to it, which must be mitigated. Future work will validate the framework on real worker-task data, replace the skill graph with pretrained
> sentence embeddings, measure user trust in a study, and test mitigations such as alternative evidence routes.

---

## 4. Table II: what to keep

| Planned metric | Status |
|---|---|
| Matching accuracy / F1 | Reported (P@1, F1@5) |
| Verification precision, recall | Reported, but set by the simulated verifier (consistency check) |
| Profile update latency | Reported (0.005 ms) |
| Explainability coverage | Reported (100% by construction) |
| Precision@K, Recall@K, NDCG@K | Reported |
| Profile adaptation gain | Reported (Table C) |
| Fairness gap | Reported (Tables E-F) |
| Inference latency | Reported (15 ms) |
| Explanation faithfulness | Move to future work (trivial for an additive score) |
| User-reported trust/satisfaction | Move to future work (needs a user study) |

---

## 5. Other edits to the paper

1. Section I.1: "RQ3 ... without making the system impractical to operate" -> add the latency result.
2. Fig. 4 (timeline) can stay; mark the implementation and testing bars as completed.
3. Add Fig. 5 (dashboard screenshot): run `python app.py`, open http://127.0.0.1:5000 and screenshot it. I tested the app only through Flask's test client, not in a browser, so check how it looks.
4. Add `results/fig_comparison.png`, `fig_learning_curve.png` and `fig_inflation_sweep.png` as figures. Add the dashboard figure and a link to your code repository.
5. Fix the citation and reference problems listed earlier (shifted Table I numbers, duplicate references [4]/[8] and [5]/[15], uncited [6], [7], [18], section order, and the unsupported "improves fairness" claim about [5]).
