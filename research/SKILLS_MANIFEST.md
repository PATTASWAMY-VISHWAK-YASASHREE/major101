# Skills Manifest — Cross-Machine Sync

**Purpose:** Hermes skills live in the machine's profile (`~/.hermes/skills/` — on Windows: `%LOCALAPPDATA%\hermes\skills\`), NOT in this repo. A git clone loses them. This manifest is how a fresh agent on the Ultra 9 (or any machine) restores the working set.

**Verified mechanics (from hermes-agent docs + this machine, 2026-08-31):**
- A skill = a plain folder containing `SKILL.md` (+ optional `scripts/`, `references/`, `assets/`, `templates/`).
- Fresh `hermes` install ships ALL bundled skills automatically (the installer handles it — `curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash`, or the Windows path from the docs install page).
- Therefore: bundled skills need NO sync — install Hermes and they're there.
- Only machine-specific or third-party skills need copying. This session used 5 stacked skills, all of which are bundled.

---

## Method A (preferred): fresh install covers most of it

```bash
# 1. Install Hermes on the Ultra 9 (brings every bundled skill):
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
#    Windows native: see https://hermes-agent.nousresearch.com/docs/getting-started/installation

# 2. Verify a skill is available:
hermes chat --toolsets skills -q "list available skills"   # or check skills_list in-session
```

## Method B: copy the folder (for anything not bundled)

Skills are plain directories. To copy from this i5 to the Ultra 9:

```powershell
# On the i5 (source) — bundle the needed skills into the repo-adjacent transfer:
$src = "$env:LOCALAPPDATA\hermes\skills"
$dst = "D:\skill_transfer"       # USB / OneDrive folder
# ...copy each folder from the table below...

# On the Ultra 9 (target) — drop into the profile:
$home9 = "$env:USERPROFILE\.hermes\skills"   # default $HERMES_HOME on a fresh install
# ...paste folders here, restart hermes...
```

Verify after copying: `hermes chat --toolsets skills -q "use the <skill-name> skill: what does it cover?"`

---

## The working set (verified present on the i5, 2026-08-31)

| Skill | Why this project needs it | Bundled? | Copy path |
|---|---|---|---|
| `citation-management` | BibTeX/PMID verification — built the consolidated verified bib; paper #2 citations | likely | `skills\citation-management\` |
| `scientific-writing` | Manuscript drafting/revision discipline for both papers | likely | `skills\scientific-writing\` |
| `research` | Background-agent research pattern (used for the dataset sweep) | likely | `skills\research\` |
| `pydicom` | DICOM parsing for Brain-Tumor-Progression (companion glioma dataset) | likely | `skills\pydicom\` |
| `statistical-analysis` | Bootstrap CIs, honest intervals for Phase 2 (n=40 → intervals matter) | likely | `hosts>skills\statistical-analysis\` |
| `literature-review` | Related-work section for paper #2 | likely | `skills\literature-review\` |
| `venue-templates` | Journal/conference formatting when paper #2 gets a target venue | likely | `skills\venue-templates\` |
| `peer-review` | Pre-submission self-review of both manuscripts | likely | `skills\peer-review\` |
| `bids` | Longitudinal MRI organization conventions (BIDS naming for timelines) | likely | `skills\bids\` |
| `matplotlib` | Figures for paper #2 (tumour trajectories, pheromone-field overlays) | likely | `skills\matplotlib\` |
| `markdown-mermaid-writing` | Architecture diagrams in docs (colony diagrams) | likely | `skills\markdown-mermaid-writing\` |

All 11 verified present on the i5 via directory check (2026-08-31). "Bundled?" is *likely* for all — meaning a fresh `hermes` install on the Ultra 9 probably already has them; **verify with `skills_list` after install and only copy what's missing** (Method B). If any skill below is missing AND was modified on the i5 (some were curated), copy the i5 version to keep the improvements.

**Session-used skills (the 5 stacked for this project's birth):** `citation-management`, `research-grants` (now historical — grants rejected), `research`, `bgpt-paper-search`, `consciousness-council`. Only the first three remain relevant.

---

## Automatic path (instead of manual copying): symlink

The user asked about symlinks. Reality check (Windows): symlinks to a directory outside the profile work (mklink /J junction — no admin needed) but **tie both machines to the synced folder permanently** and confuse the curator with cross-machine state. Recommendation: **don't symlink**. Copy once, or rely on the fresh install. If you insist:

```powershell
# junction from a OneDrive-synced skills folder into the live profile
New-Item -ItemType Junction -Path "$env:USERPROFILE\.hermes\skills\citation-management" -Target "D:\OneDrive\hermes_skills\citation-management"
```

…repeat per skill. But a fresh install + copy-missing is cleaner and curator-safe.

---

## What a fresh agent on the Ultra 9 should do (add to AGENTS.md §9)

1. `hermes setup` → fresh install brings bundled skills
2. Run `skills_list` in-session (or the verify command above) against the manifest table
3. Any missing skill → copy from i5 via the transfer folder (Method B) — folders are self-contained
4. Only then proceed with the mission kit (§9 of AGENTS.md)

**Manifest last verified:** 2026-08-31 on the i5. Update this file whenever a new skill becomes project-relevant or a machine's skill set changes.
