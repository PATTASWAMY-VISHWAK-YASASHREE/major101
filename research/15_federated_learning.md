# Category 15: Federated & Privacy-Preserving Learning

**Status: 🔬 RESEARCH ONLY — no implementation yet**

> **Purpose:** Understand federated learning for medical imaging. Relevant only if
> the project scales to multi-institutional data. Out of scope for this prototype.

---

## 15.1 What Is Federated Learning?

```
Federated Learning:
1. Multiple hospitals each train a local model on their own data
2. Only model weights (not patient data) are shared
3. A central server aggregates weights → global model
4. Global model weights are sent back to each hospital
5. Repeat

Result: Multi-institutional model trained without sharing patient data.
```

**Why relevant:** Brain tumour data is highly sensitive. Hospitals cannot share
NIfTI volumes due to HIPAA/GDPR/IRB constraints. Federated learning lets them
collaborate without violating privacy.

---

## 15.2 Federated Learning for BraTS

```
BraTS challenge federated setup (conceptual):
- Each participating institution has its own BraTS data subset
- Each institution trains ResNet3D locally
- Weight updates are shared (not patient data)
- Central server aggregates via FedAvg

Result: Global BraTS model trained on ~10,000 cases across 50 institutions
```

**Status:** Active research area. Several federated BraTS approaches published
(2022-2025). Not yet standard practice.

---

## 15.3 For Our Project

| Aspect | Status |
|---|---|
| **Single-institution prototype** | No federated learning needed |
| **Multi-institutional deployment** | Federated learning becomes relevant |
| **Current dataset (IBSR, single site)** | ❌ Not applicable |

**Verdict:** Federated learning is out of scope for this project. It is an
engineering concern for the deployment phase, not the research phase.

---

## 15.4 Future Consideration

If this project moves toward clinical deployment across multiple hospitals:

1. **Federated BraTS** approach: Each hospital trains locally on their BraTS data,
   shares model weights only.
2. **Secure aggregation:** Use secure multi-party computation to ensure no hospital
   can reverse-engineer another's data from the shared weights.
3. **Differential privacy:** Add noise to shared weights to provide mathematical
   privacy guarantees.

**Not in scope for this prototype.**
