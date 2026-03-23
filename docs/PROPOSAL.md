# Radiology Pre-Screening Subnet (TB First)

## Bittensor Subnet Ideathon Proposal

### Summary

We propose a Bittensor subnet focused on **radiology pre-screening AI**, starting with **TB detection** and expanding to additional chest imaging tasks (e.g., silicosis and related abnormalities). The subnet is designed to drive the development of **state-of-the-art (SOTA)** models for this domain through a measurable, incentive-driven competition.

The core thesis is simple: this is a high-value, high-demand domain with clear evaluation targets, continuous data generation, and strong need for generalization. These properties make it well-suited for Bittensor’s miner-validator incentive model.

---

## Why This Subnet Should Exist

Radiology systems in many regions face a structural gap between imaging volume and radiologist capacity. This creates delays in review and prioritization. Pre-screening AI can help triage and prioritize cases for faster human review.

This is also a commercially attractive domain:

* multiple AI companies are actively approaching medical data archiving companies to obtain training data for radiology AI,
* which signals strong market demand and urgency,
* but most of these efforts are centralized and closed.

Our subnet aims to prove that a decentralized incentive network can compete with — and potentially outperform — centralized AI vendors by accelerating model improvement through open competition.

---

## Core Subnet Objective

Build a Bittensor subnet that produces **SOTA radiology pre-screening models** by rewarding miners for model performance on data generated **after** their training period.

This is not a static benchmark subnet. It is a **continuous model improvement system**.

---

## Incentive Mechanism (Core Design)

### What Miners Produce

Miners do **not** merely return predictions as the primary output.
The miner’s output is a **trained model**.

Miner workflow:

1. Train a radiology pre-screening model.
2. Publish/version the model artifact (e.g., Hugging Face).
3. Host the model on **Chutes (SN64)** for standardized inference access.
4. Submit model metadata (model version, timestamp, task compatibility).

### What Validators Do

Validators evaluate models running on Chutes using standardized requests and scoring logic.

The key requirement is **temporal evaluation integrity**:

* validators evaluate each submitted model on data that the model **could not have used during its training period**,
* and rewards are allocated based on measured performance.

### Reward Mechanism: Winner-Takes-All (Important)

This subnet uses a **winner-takes-all** reward mechanism.

At each evaluation cycle:

* submitted models are scored by validators,
* models are ranked by performance,
* and the **top-performing miner receives the reward allocation** for that cycle (subject to subnet implementation details).

Why winner-takes-all is important in this subnet:

* it creates maximum competitive pressure to reach **SOTA** performance,
* it strongly discourages low-effort participation,
* and it pushes miners to focus on real, sustained model improvement rather than incremental farming strategies.

This is intentionally aggressive by design to optimize for frontier performance in a highly valuable domain.

### Time-Shifted Evaluation (Important Clarification)

The evaluation data is **not required to be secret or hidden**.

The key mechanism is **time delay**, not secrecy:

* if a model is trained and submitted at time **T**,
* it is evaluated on data generated during a later window (e.g., **T+M**),
* where **M** is the evaluation delay window.

Miners can still use previously evaluated data for future training.
That is acceptable and expected.

What matters is:

* a model is only scored on data that did not exist (or was not yet available) during that model’s training period.

This rewards **generalization**, not static benchmark tuning.

---

## Evaluation Delay Window (To Be Determined)

The subnet will test and optimize an evaluation delay window in the range of:

* **1 day to 7 days**

The best window will be selected based on:

* data availability cadence,
* operational practicality,
* and how effectively the delay prevents short-cycle overfitting while preserving rapid iteration.

This parameter is a core part of the subnet design and will be empirically tuned.

---

## Why Bittensor Fits This Use Case

This domain is a strong fit for Bittensor because it has the key properties required for a high-quality incentive network:

* **Measurable performance** (e.g., sensitivity, specificity, false-negative behavior, calibration)
* **Continuous data generation** (new imaging and reports over time)
* **Strong value of generalization** (future-case performance matters more than benchmark performance)
* **High-value market demand** (multiple companies already competing for access to data in this domain)

Bittensor provides the mechanism to coordinate many independent teams and reward only those that measurably improve model quality.

---

## Why This Can Beat Centralized AI Vendors

Centralized AI vendors typically rely on a single internal research pipeline. In contrast, this subnet creates:

* parallel model development from many independent teams,
* transparent competition on measurable outcomes,
* continuous retraining and iteration,
* and open model publication / auditability.

This structure can increase the rate of improvement and make model performance more contestable and evidence-driven.

---

## Data and Partnership Model (High-Level)

The subnet relies on partnerships with medical data archiving companies for redacted/de-identified imaging and associated reports/metadata (as permitted).

This is foundational because:

* data quality and continuity determine subnet quality,
* and ongoing data flow enables continuous evaluation and improvement.

The subnet begins with radiology pre-screening for TB and is designed to expand to additional disease categories over time.

---

## Scope and Positioning

This subnet is for **pre-screening / triage support**, not diagnosis replacement.
Human clinicians (radiologists) remain the final decision-makers.

---

## Closing Statement

This subnet is designed to build **SOTA radiology pre-screening models** through a clear, technically grounded incentive mechanism:

* miners produce trained models,
* models are hosted on Chutes,
* validators evaluate those models on future-period data the models could not have used at training time,
* and a **winner-takes-all** reward structure directs incentives to the best-performing miner each cycle.

We believe this is one of the strongest real-world applications of Bittensor’s incentive design: a high-demand domain, clear utility, continuous data, and a credible path to outperform centralized AI development through decentralized competition.
