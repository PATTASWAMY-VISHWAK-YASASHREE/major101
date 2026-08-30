# Consciousness Council — Strategic Deliberation Report

**Project:** BraTS CT+MRI Multimodal Brain Tumour Classification (major101)
**Date:** 2026-08-30
**Context:** Tier-3 college, no budget, shitty systems, public datasets only, RTX 2050 4GB / 3GB RAM

---

## The Question

Given the actual state of this project — a working MRI-only classifier with honest modest results (AUROC 0.7672), a completed 117-paper literature review, partially cleaned citations, a vision of CT+MRI fusion but a fundamental blocker (no paired public dataset in BraTS), and a tier-3 college with zero budget and poor compute — what should the team PRIORITIZE? Publications? Grants? Code? Something else?

---

## 🎭 THE PRAGMATIST

**Position:** Ship the MRI-only paper first, then build a CT+MRI fusion demo on whatever public paired data actually exists — but stop pretending BraTS is where the CT+MRI story will be told.

**Reasoning:**

Here is what we have, stripped of aspiration:

1. **One finished, defensible MRI-only classifier.** AUROC 0.7672 on locked test. Balanced accuracy 0.5261. The label is an ET-derived proxy — not WHO grade. The pipeline is data-audited (882/882 files verified, 876 unique cases, subject-disjoint splits, 12/12 contract tests passing). This is honest, reproducible, and citable. It is NOT clinically validated, and the panel report says exactly that.

2. **The CT+MRI vision has a data problem.** The project documents say "no paired CT+MRI public dataset exists." This was TRUE for BraTS. But my search found: IEEE DataPort glioblastoma benchmark (50 patients with paired CT+MRI from Radiopaedia), Kaggle Brain Tumor Multimodal (CT+MRI images), Zenodo brain metastases (185 studies, 40 patients, CT+MRI), and a PMC paired CT-MRI dataset (18 patients, T1/T2/CT). These are small, messy, and not BraTS — but they EXIST. The "no dataset" claim needs updating to "no large, curated, BraTS-quality paired dataset exists."

3. **Synthetic CT from MRI is real but not-for-diagnosis.** Multiple papers (2024-2026) demonstrate MRI-to-CT synthesis using GANs and diffusion models. This is validated for radiotherapy planning (dose calculation), not for diagnostic classification. Using synthetic CT to train a fusion classifier is a research exercise, not a clinical contribution. It might be worth doing as a "what if" ablation, but it must be labelled as synthetic.

4. **What can actually ship within 3-6 months:**
   - A paper: "MRI-only glioma grade-proxy classification on BraTS 2024: a memory-bounded, data-audited baseline" — honest, with limitations clearly stated. Target: a mid-tier journal (Scientific Reports, PLOS ONE, or a neuro-oncology/ML conference).
   - A demo: extend the existing pipeline to accept CT input, train on one of the small public paired datasets (Kaggle multimodal or IEEE DataPort), show what happens — even if the numbers are weak. The contribution is the pipeline + the honest evaluation, not breakthrough accuracy.
   - A benchmark release: preprocessed public CT+MRI datasets with documented pipeline, so other researchers can build on it.

**Key Risk:** Spending 12 months trying to build a CT+MRI fusion that looks like it should work on BraTS-scale data when the public data doesn't support it, and then having nothing citable to show for it.

**Surprising Insight:** The "no dataset" blocker is real for BraTS but false in general. The project's entire CT+MRI research track was gated on a claim that is only partially true. Updating this claim and finding the actual available data is the single highest-leverage action right now.

---

## 🎭 THE STRATEGIST

**Position:** The publication strategy should be honest-modest-first, then pivot the grant story toward "accessible multimodal medical AI for resource-limited settings" rather than "state-of-the-art fusion accuracy."

**Reasoning:**

**What you actually have for publications:**

- A data-audited, reproducible MRI-only classifier on a major benchmark (BraTS 2024). The honesty about the ET-derived proxy label and the modest performance is actually a STRENGTH for certain venues — it shows scientific integrity. Many papers in this field overclaim; a paper that says "our AUROC is 0.7672 and here's exactly what that means and doesn't mean" stands out.
- A comprehensive literature review (117 papers, 19 categories, 2013-2026). This alone is citable as a review — but only if the citations are clean. The fabricated PMIDs in cat4/5/8 must be excised before any review submission. The verified_foundational_references.bib + consolidated_verified_references.bib give you a clean foundation.
- A resource-constrained engineering story: "here's a 3D CNN that runs on 4GB VRAM and 3GB RAM." This is publishable as a methodology/engineering contribution.

**Publication venues in order of fit:**

1. **Mid-tier journals (Scientific Reports, PLOS ONE, BMC Medical Imaging, Computers in Biology and Medicine):** Accept honest, methodologically sound work with modest results. The MRI-only paper fits here if the framing is "rigorous baseline + data audit + reproducibility" rather than "state-of-the-art classification."
2. **Conference workshops (MICCAI workshops, ISBI workshops):** Lower barrier, faster turnaround, good for the fusion demo and pipeline release.
3. **Top-tier journals (Medical Image Analysis, IEEE TMI):** Would require substantially stronger results or a truly novel methodological contribution. The current MRI-only results don't qualify alone. The fusion paper might if it establishes the first open benchmark and generates genuine interest — but don't target this until you have the data.

**Grant strategy — what fits a tier-3 college with no budget:**

- **NIH R21:** Up to $275K over 2 years, no preliminary data required, 6-page research strategy. This is the most realistic NIH mechanism. The specific aims draft I wrote targets this exactly. The "no preliminary data required" clause is critical — your current MRI results serve as feasibility demonstration but aren't mandatory. However, NIH requires an NIH-specific format, US-institutional PI, and the review criteria (Significance, Investigator, Innovation, Approach, Environment) are harsh on "Environment" for under-resourced institutions. You would need to partner with a US institution or find a way to make the "Environment" criterion work — the open-source, accessible-AI angle helps here.

- **NSF (CISE or ENG):** NSF funds computer science and engineering, not clinical research. A proposal framed as "resource-constrained multimodal deep learning for medical imaging" with strong intellectual merit (the technical contribution) and broader impacts (democratising AI for low-resource settings) could fit NSF CISE. The broader impacts angle is genuinely strong here — it's not tacked-on; it's the core motivation. But NSF requires US institution, and the "Results from Prior NSF Support" section would be empty for a first-time applicant.

- **ABTA (American Brain Tumor Association):** Medical Student Summer Fellowship ($3,000) — if there's a medical student on the team. CNS Metastasis Research Grant ($50,000/year) — requires a lead mentor at the same institution doing brain tumor research, which may not exist at a tier-3 college.

- **India-specific:** DST (Department of Science and Technology), SERB (Science and Engineering Research Board), ICMR (Indian Council of Medical Research) — these are the realistic domestic funding sources. Their requirements and success rates vary widely. The story of "open-source medical AI for resource-limited settings" aligns well with DST/SERB priorities around accessible technology.

- **The brutal truth:** Most international grants require a US/EU institution, a track record, and preliminary data that tells a story of prior success. A tier-3 college with no prior grants and no budget is at a severe disadvantage for competitive international funding. The realistic grant path is: (a) domestic Indian funding (DST/SERB/ICMR), (b) small foundation grants (ABTA, Lilabean), (c) in-kind collaboration with a better-resourced institution where the partnership brings the institutional credibility.

**Key Risk:** Writing a grant proposal that's honest about the limitations but honest-modest doesn't fit grant culture, which rewards ambitious claims. The tension: a grant that overclaims gets rejected for lack of credibility; a grant that's too honest about limitations looks weak. The sweet spot is "ambitious goal, honest current state, clear path."

**Surprising Insight:** The resource-constrained angle is not a weakness to hide — it's a differentiation strategy. Most medical AI research assumes infinite compute and proprietary data. A project that deliberately targets commodity hardware and public data has a genuine broader impacts story that few competitors can match.

---

## 🎭 THE FUTURIST

**Position:** CT+MRI fusion as a standalone classification problem is a shrinking research frontier; the longer-term value is in (a) foundation models that learn from all modalities jointly, (b) synthetic modality generation validated for specific clinical tasks, and (c) the resource-constrained angle which becomes MORE valuable as models get bigger and hardware gets more expensive.

**Reasoning:**

The field is moving in directions that make the current CT+MRI fusion vision look somewhat backward-looking:

1. **Foundation models for medical imaging** (MONAI, MedSAM, various vision-language models) are trained on massive multi-modal datasets and can perform segmentation, classification, and VQA from a single model. The research question "should we concat CT and MRI features?" may be answered by "just use the foundation model" within 3-5 years.

2. **Synthetic modality generation** is getting serious. The SynthRAD2025 grand challenge focused on MRI-to-CT conversion. Diffusion models (VS-DDPM, MC-IDDPM) are producing clinically acceptable synthetic CTs for radiotherapy. If synthetic CT becomes good enough for specific clinical tasks (not all tasks), the "we need real paired CT+MRI" problem partially dissolves — but only for the tasks where synthetic CT has been validated, and that validation is task-specific and doesn't automatically transfer to classification.

3. **BraTS itself is evolving.** BraTS 2024 focused on post-treatment glioma segmentation. Future BraTS challenges may incorporate additional modalities or tasks. The "BraTS is MRI-only forever" assumption may not hold, but it's also not something to wait for.

4. **Where this project fits in 3-5 years:** If the team establishes a genuine open benchmark for CT+MRI brain tumour classification on public data — even with small datasets and modest results — that benchmark becomes a reference point. The resource-constrained, open-science angle becomes MORE valuable as the field concentrates on large models and proprietary data. The team's niche is "we did rigorous, honest, accessible work on public data when everyone else was chasing SOTA on private data."

**Key Risk:** Building a CT+MRI fusion paper that gets published in 2027 and is already outdated because foundation models made the specific fusion architecture irrelevant. The hedge is to frame the contribution as "the first open benchmark + honest evaluation" rather than "our fusion architecture is the best."

**Surprising Insight:** The most defensible long-term contribution isn't the fusion accuracy — it's the benchmark and the pipeline. Benchmarks outlive architectures. A well-documented, reproducible benchmark on public data that other researchers use is a citation magnet that doesn't depend on staying state-of-the-art.

---

## 🎭 THE OUTSIDER

**Position:** You're overcomplicating the fusion story and under-selling the reproducibility story. The thing that's actually rare and valuable here isn't the model architecture — it's the data auditing, the leakage prevention, the locked-test protocol, and the honesty about what the results mean.

**Reasoning:**

I'm not a medical imaging person. I look at this project as a generic ML engineering effort. Here's what jumps out:

1. **The data work is the real contribution.** Deduplicating 994 rows to 876 cases. Rejecting conflicting labels. Grouping repeated acquisitions by subject to prevent leakage. Validating 882 files for shape, dtype, and non-finite values. Writing 12 contract tests. Running a locked test exactly once. Calibrating on development data only. This is the kind of rigorous ML engineering that the field desperately needs and rarely sees. Most published medical AI papers don't do any of this — they split randomly, don't check for leakage, and report accuracy on a test set they may have indirectly tuned on.

2. **The fusion vision is over-engineered for the data reality.** The architecture diagrams show per-modality 3D ResNet encoders, cross-modal attention, temporal transformers, survival heads — a full production system. But the available public data is 50 patients (IEEE DataPort) or 40 patients (Zenodo). You can't train a multi-branch 3D CNN with cross-modal attention on 50 cases. The gap between the vision and the data is enormous, and the project documents mostly acknowledge this in the research sections but then continue planning as if the data will appear.

3. **"No budget, tier-3 college" is a constraint but also an identity.** The project could lean into this identity hard: "We are a team at a resource-limited institution doing rigorous, open, reproducible medical AI on public data with commodity hardware." That's a story. It's not the story every lab tells, and that's the point. It's honest, it's differentiated, and it has genuine broader impacts value.

4. **The citation integrity work is under-appreciated.** The audit found fabricated PMIDs in subagent-generated bib files. This is a systemic problem in AI-assisted research — the subagents that wrote the literature review invented plausible PMIDs that resolved to wrong papers. The fact that this was caught, documented, and corrected (cat17/18/19 verified, cat4/5/8 flagged, consolidated bib created) is worth a methods note in any paper that uses these references.

**Key Risk:** Continuing to plan the fusion architecture as if the data problem is temporary and will be solved, when the realistic path is working with the small public datasets that exist and being honest about their limitations.

**Surprising Insight:** The most citable, most valuable thing in this repository might be the data-quality report and the contract tests — not the model. A paper that says "here's how we audited our medical imaging data, here are the tests we wrote to prevent leakage, and here's what we found" could be genuinely useful to the community. That's a paper that doesn't need ANY fusion results.

---

## 🎭 THE MINIMALIST

**Position:** Stop planning. Pick ONE of three paths and execute it for 8 weeks. The current state is analysis paralysis with a lot of research documentation and no clear next action.

**Reasoning:**

Here's what to STOP doing:

1. **Stop planning the full fusion architecture.** The diagrams with 5 modality encoders, cross-modal attention, temporal transformers, and survival heads are a vision document, not a plan. No one is building that system. It's cluttering the vision because it suggests the work is bigger than it is.

2. **Stop treating the "no dataset" claim as absolute.** You found paired CT+MRI datasets. Update the claim. The datasets are small and not BraTS-quality, but they exist. Work with what exists.

3. **Stop accumulating research categories.** You have 19 research categories with cross-references. Most of them are documentation of what you READ, not what you're DOING. Documentation is good up to a point; beyond that it's procrastination in markdown format.

4. **Stop treating the MRI-only result as a stepping stone to something better.** It's a finished, defensible, citable result. Package it. Write the paper. Submit it. The paper doesn't need to be Nature-level — it needs to be honest and complete.

Here's what to START doing — pick ONE:

**Option A — Publish MRI-only:**
- Write the paper using the panel report as the base
- Target: Scientific Reports or PLOS ONE or a workshop
- Timeline: 4-6 weeks to write, 2-3 months to review
- Deliverable: One published paper, honest about limitations
- This is the safest path to a citable output

**Option B — Build CT+MRI demo:**
- Download IEEE DataPort or Kaggle multimodal dataset
- Extend pipeline to accept CT
- Train MRI-only baseline on the CT+MRI dataset, then add CT fusion
- Report honestly whatever happens
- Timeline: 6-8 weeks
- Deliverable: A fusion demo paper (workshop or low-tier journal) + preprocessed dataset release
- This is the path that advances the fusion vision, but with weak results likely

**Option C — Write the benchmark paper:**
- Focus on the data auditing + reproducibility story
- "Rigorous data auditing and leakage-prevention protocol for brain tumour MRI classification: a BraTS 2024 case study"
- Target: a methods/reproducibility venue
- This leverages the thing that's genuinely rare (the data work) without needing strong model results

**Option D — Write a grant proposal:**
- Use the specific aims draft I've prepared (aimed at NIH R21 or equivalent)
- Find a domestic Indian funding source (DST, SERB, ICMR) or a small foundation (ABTA)
- This is the longest road but potentially the highest payoff if it brings funding
- Realistic timeline: 3-6 months from writing to decision

**What I'd pick:** Option A first (publish MRI-only), running in parallel with the earliest steps of Option B (download and verify one public paired dataset). The MRI paper gives you a citable output and forces you to finish something. The dataset verification tells you whether Option B is actually feasible. Don't decide on B until you've seen the data.

**The smallest viable next step right now:** Pick one public paired CT+MRI dataset (IEEE DataPort is the most purpose-built for brain tumour CT+MRI), download it, run it through a basic verification script (file counts, shapes, any labels available), and write a one-page report on what's actually in it. That takes a day or two and immediately upgrades the team's knowledge from "no dataset exists" to "here's what the dataset contains and what it would take to use it."

---

## ⚖️ COUNCIL SYNTHESIS

### Points of Convergence

1. **The MRI-only result is finished and should be published.** All five members agree that the data-audited, reproducible MRI classifier with honest results is a citable contribution that should be written up. The honesty about the ET-derived proxy label and modest performance is a feature, not a bug, for certain venues.

2. **The "no paired CT+MRI dataset" claim is outdated.** The Futurist, Outsider, and Pragmatist all flagged this. The claim was true for BraTS but false in general. Public paired datasets exist (IEEE DataPort, Kaggle, Zenodo, PMC). This claim must be updated before any publication or grant that relies on it.

3. **The resource-constrained angle is a genuine differentiator.** Every member saw value in the "commodity hardware + public data + rigorous engineering" story. This is the team's natural broader impacts angle and should be central to any grant or publication framing.

4. **The data auditing is more valuable than the model.** The Outsider and Pragmatist both highlighted that the data-quality work, leakage prevention, and contract tests are rare and citable — possibly more so than the classification results.

5. **Analysis paralysis is real.** The Minimalist was bluntest about this: 19 research categories, architecture diagrams for systems that don't exist, gating conditions that may be based on outdated claims — the project is over-documented and under-executed.

### Core Tension

**Honest-modest vs. grant-competitive.**

Grants reward ambitious claims. A tier-3 college with no budget and modest results is at a structural disadvantage for competitive international funding. The team must decide: do they write grants that are ambitious enough to have a chance (but risk looking overclaimed), or honest enough to be credible (but risk looking weak)? There's no clean answer. The Pragmatist and Strategist both see the tension. The best hedge is: publish the honest MRI paper first (establishes credibility), THEN write a grant that uses the published paper as preliminary data and makes appropriately ambitious claims from a position of demonstrated capability.

### The Blind Spot

**Nobody addressed: who is the PI, and what is their track record?**

Grant applications live or die on the Investigator criterion. A first-time PI at a tier-3 college with no prior grants, no publications yet (from this work), and no institutional research infrastructure is a hard sell for ANY competitive grant — NIH, NSF, DST, SERB, whatever. The specific aims draft I wrote doesn't address this because I don't know the PI's situation. This is the single most important question for the grant path: is there a PI with a track record, or is this a team that needs to build credibility through publications before grants are realistic?

### Recommended Path

**Phase 1 (now → 8 weeks): Publish + verify.**

1. Write and submit the MRI-only paper (honest, data-audited, limitations clearly stated). Target: mid-tier journal or workshop.
2. Download and verify ONE public paired CT+MRI dataset (IEEE DataPort recommended). Write a one-page data report. Decide whether Option B (fusion demo) is feasible based on what the data actually contains.

**Phase 2 (8 → 16 weeks): Fusion demo OR grant.**

- If the public dataset is workable: build the CT+MRI fusion demo, write a second paper (workshop or low-tier), release the preprocessed dataset as a benchmark.
- If the public dataset is too small/messy: focus on the reproducibility/data-auditing paper angle instead, and start drafting a grant proposal using the published MRI paper as preliminary data.

**Phase 3 (16+ weeks): Grant applications.**

- Use the published paper(s) as credibility. Target domestic Indian funding (DST/SERB/ICMR) as the most realistic path. Consider US NIH R21 only if there's a US institutional partner or co-PI.

### Confidence Level

**Medium.** The convergence around "publish MRI-first, verify the dataset claim, stop over-planning" is strong across all five perspectives. The uncertainty is in the grant path — the team's institutional context and PI track record are unknown variables that dramatically affect grant feasibility.

### One Question to Sit With

**"If we had to ship one citable thing in 8 weeks, what would it be — and are we willing to accept that it won't be a breakthrough, just a complete, honest, reproducible contribution?"**

The council's answer is: the MRI-only paper with the data audit. That's the thing that exists right now, fully done, ready to write up. Everything else is contingent on data that may or may not work out.

---

*Report generated by the Consciousness Council: Pragmatist, Strategist, Futurist, Outsider, Minimalist.*
