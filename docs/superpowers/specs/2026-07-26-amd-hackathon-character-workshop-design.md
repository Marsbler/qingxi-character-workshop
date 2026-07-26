# Design: AMD AI DevMaster Hackathon — Original Fantasy Character Workshop

**Date:** 2026-07-26  
**Track:** Track 1 — Multimodal Content Creation Tools  
**Status:** Draft for user review (design frozen pending approval)  
**Deadline:** 2026-08-06 23:59 Beijing time (~12 days from design date)

---

## 1. Problem and goal

### 1.1 Goal

Build a **lightweight, high-performance multimodal creation tool** that runs on **AMD Radeon GPU + ROCm**, delivered as a **clean web workbench**:

- **Input:** text character brief; optional reference image (sketch or photo).
- **Output (P0, must ship):** structured lore + anime-style portrait + six-affinity radar + downloadable character card poster.
- **Output (P1, soft):** 2–4 second ability showcase video (image-to-video); failure must not block the card.

### 1.2 Non-goals (explicit)

- Multi-session chat product (Doubao/DeepSeek clone).
- Large RAG / full lore bible / novel-length worldbuilding.
- Training foundation models or large SFT corpora.
- Using third-party copyrighted IP (e.g. 罗小黑 / Luo Xiaohei) for branding, training data, or character replication.
- Core features that only work via closed-source online APIs.

### 1.3 Success criteria (competition)

| Criterion | Target |
|-----------|--------|
| Rules fit | Track 1 multimodal workflow; ≥1 core task (txt2img / img2img / I2V) |
| Hardware | Critical inference on AMD Radeon + ROCm locally |
| P0 demo | End-to-end character card on Radeon Cloud (or equivalent AMD env) |
| P1 demo | Real short video **or** documented graceful degradation |
| Submission | PDF + source + 3–5 min demo video + PPT/poster; PR to official fork |
| IP | Original thin world only; no infringing training assets |

---

## 2. Product decisions (locked)

| Decision | Choice |
|----------|--------|
| Approach | **Scheme 2:** Card hard-guarantee + video soft-guarantee |
| World | **Original thin shell** (not third-party IP); ~1 page of rules |
| UI | **Workbench** (left input / right results); concise and polished; **full pipeline features retained** |
| Chat | No multi-session; optional “revise this result” on current job only |
| Training | **Default zero training**; optional style LoRA as P2 only |
| Dev workflow | **Hybrid:** code/UI locally; heavy compute, training (if any), and demo recording on **Radeon Cloud** |

---

## 3. UI / UX

### 3.1 Layout

Single-page workbench:

- **Top bar:** logo, app name, one-line world tagline, “Local · ROCm” badge.
- **Left — Create:** multiline brief, reference upload/clear + thumb, optional affinity preference (auto or pick), primary **Generate character card**, secondary **Revise from result**.
- **Right — Results:** card poster (portrait + radar + text blocks), lore panel, download card / copy lore, **Generate ability animation** + video area or error, step progress.

### 3.2 Feature inventory (do not cut pipeline; cut product chrome only)

| Feature | Priority |
|---------|----------|
| Text brief | P0 |
| Reference image | P0 |
| Optional affinity control | P0 |
| Generate character card | P0 |
| Portrait + structured lore + affinity viz + downloadable poster | P0 |
| Step progress + readable errors + retry | P0 |
| Footer: AMD Radeon + ROCm local inference | P0 |
| Generate ability animation (2–4s) | P1 |
| Revise from current result | P1 |
| Multi-session, accounts, plugin marketplace | Out of scope |

### 3.3 Primary demo path

1. Enter brief (+ optional ref) → Generate card.  
2. Progress completes → poster + lore → download.  
3. Generate ability animation → video or friendly failure.  
4. Optional one-line revise → regenerate image + card (video not auto-rerun).

### 3.4 Implementation stack (UI)

- **Default:** Gradio Blocks + light custom CSS (fast, enough polish).  
- **Fallback:** same backend + minimal Gradio if custom CSS slips.  
- Avoid full Next.js multi-session clone within the timebox.

---

## 4. System architecture

### 4.1 Diagram

```
Presentation (Gradio workbench)
        │
        ▼
Orchestrator (job state machine, progress, errors)
        │
        ├─► world/          static thin lore config
        ├─► llm_role/       text → character.json
        ├─► image_gen/      txt2img / ref-guided img
        ├─► card_compose/   Pillow poster
        └─► video_gen/      optional I2V
        │
        ▼
outputs/{job_id}/
```

### 4.2 Modules

| Module | Responsibility | Runs on |
|--------|----------------|---------|
| `world/` | Affinities, prompts fragments, IP bans | CPU |
| `llm_role/` | Structured JSON + image/motion prompts | Radeon local LLM |
| `image_gen/` | Portrait generation ± reference | Radeon Diffusers |
| `card_compose/` | Radar + layout → `card.png` | CPU |
| `video_gen/` | Portrait → short mp4 | Radeon (P1) |
| `orchestrator/` | Job lifecycle, degradation | Same host as UI |
| `ui/` | Workbench | Same host; optional `rc-tunnel` |

### 4.3 Character JSON contract

```json
{
  "name": "string",
  "one_liner": "string",
  "appearance": "string",
  "personality": "string",
  "backstory": "string",
  "primary_affinity": "one of six",
  "affinities": { "形": 0, "念": 0, "生": 0, "质": 0, "空": 0, "时": 0 },
  "spirit_domain": "string",
  "ability_showcase": "string",
  "image_prompt": "string",
  "image_negative": "string",
  "motion_prompt": "string"
}
```

Affinity **display names** come from `world.yaml` (IDs stable in code; labels localized). Values 0–100 for radar.

### 4.4 Job artifact layout

```
outputs/{job_id}/
  input.txt
  ref.png              # optional
  character.json
  portrait.png
  radar.png            # optional intermediate
  card.png
  ability.mp4          # optional
  meta.json            # timings, device, model ids, video ok/fail
```

### 4.5 Progress states

`queued → planning → imaging → composing → ready → (animating) → done`

- At `ready`, card is downloadable.  
- `animating` only after user clicks ability animation.  
- Failures set `failed_step` + message; video failure keeps `ready`.

### 4.6 User paths

**Path A — Card (P0):** text [+ ref] [+ affinity] → LLM JSON → image → compose → `ready`.

**Path B — Video (P1):** existing `ready` job → I2V → mp4 or error.

**Path C — Revise (P1):** current JSON + edit instruction → LLM patch → re-image → re-compose (no auto video).

### 4.7 Error / degradation policy

| Failure | Behavior |
|---------|----------|
| Invalid LLM JSON | One repair retry; then clear error |
| Image OOM | Lower resolution/steps; then fail card with message |
| Ref adapter unavailable | Text-only generation; UI notes limitation |
| I2V fail/timeout | Keep card; show error in video panel |
| Non-ROCm device | Detect at startup; document AMD requirement for submission |

### 4.8 Process model

Single process on one GPU instance:

- Lazy-load models (load SD on first image; load I2V only for video).  
- Prefer sequential VRAM use (free SD before I2V when needed).  
- Gradio on `127.0.0.1:7860`; optional Radeon Cloud `rc-tunnel` for public demo URL.  
- Submission does **not** require a permanently live public URL; demo video is the judge-facing proof.

---

## 5. Local vs Radeon Cloud

### 5.1 Official platform

- **Radeon Cloud:** https://radeon-global.anruicloud.com/  
- Guide: [Radeon-Cloud-User Guide](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/tree/main/Radeon-Cloud-User%20Guide)  
- Access: JupyterLab and/or SSH; prefer **Persistent (PVC)** storage; **Destroy** instance when idle (credits).  
- Expose web UI: `rc-tunnel expose --port <gradio_port>`.

### 5.2 Split of work

| Work | Where |
|------|--------|
| UI, orchestrator, world text, docs | Local git |
| Model download, real inference, optional LoRA train | Radeon Cloud |
| Demo recording | Must be AMD Radeon env |
| Free shared Chat APIs (Qwen/DeepSeek on platform) | Dev scratch only — **not** core production path for scoring claims |

### 5.3 Modes

| Mode | Behavior |
|------|----------|
| `MOCK=1` | Placeholder JSON/images/video for UI without GPU |
| `MOCK=0` | Real local models on ROCm |

Weights live outside git (`models/` or PVC); `models.yaml` holds paths.

---

## 6. Model selection

### 6.1 Principles

- ROCm-proven stacks first (PyTorch + Diffusers / Transformers).  
- Profiles for ~16GB and ~24GB VRAM.  
- Sequential model occupancy.  
- Swappable IDs via config.  
- Default **no training**.

### 6.2 Default stack

| Role | Primary | Fallback |
|------|---------|----------|
| LLM | Qwen2.5-7B-Instruct (local) | 3B if VRAM tight; 14B if headroom |
| Image | SDXL anime checkpoint via Diffusers | SD1.5 anime if OOM/speed |
| Reference | IP-Adapter and/or img2img; optional ControlNet later | img2img only |
| Video | CogVideoX-2B I2V **or** AnimateDiff (whichever runs on ROCm first) | Skip live video; document attempt |

### 6.3 Image defaults

- Resolution: 832×1216 or 1024×1024 portrait; OOM → ~768 edge.  
- Steps: ~30 (fast preset ~24).  
- Guidance: ~5–7 per checkpoint.  
- Enable attention slicing / VAE tiling as needed.  
- Single UI control for reference strength.

### 6.4 Video defaults

- 2–4 seconds only; user-triggered; timeout ~180–300s.  
- After card: free SD weights before loading I2V when VRAM limited.

### 6.5 Training

| Item | Policy |
|------|--------|
| Full FT / large SFT | Do not |
| Style LoRA | Optional P2 after P0 stable |
| Justification for judges | Engineering pipeline + local ROCm inference is sufficient per Track 1 rules; LoRA is quality polish, not eligibility |

### 6.6 VRAM profiles

| VRAM | LLM | Image | Video |
|------|-----|-------|-------|
| ≥24GB | 7B FP16 | SDXL + IP-Adapter | Prefer CogVideoX-2B-class |
| ~16GB | 7B quant or 3B | SDXL 768 or SD1.5 | Prefer AnimateDiff |
| <16GB | 3B / heavy quant | SD1.5 | Video optional/sample-only |

### 6.7 Compliance note on “little training”

Track 1 requires a usable multimodal tool on Radeon/ROCm, open models with engineering optimization, and local critical inference. It **allows** LoRA/ControlNet but **does not require** training. Zero-training + strong orchestration is a valid and preferred timebox strategy.

---

## 7. World content and datasets

### 7.1 Three layers

| Layer | Artifact | Required? |
|-------|----------|-----------|
| L0 World config | `configs/world.yaml`, `prompts/*` | **Yes** |
| L1 Eval/demo set | `data/eval/cases.jsonl` (20–50) | Strongly yes |
| L2 Style LoRA set | 50–200 legal images + captions | Optional P2 |

### 7.2 L0 thin world (complete in hours, not days)

Must define:

- World name + one-line tagline.  
- Six affinities: name, one-line rule, visual keywords.  
- Spirit-domain concept under **original naming** (no protected proper nouns from existing franchises).  
- IP policy string in system prompt.  
- Art style guide + avoid list.  
- 1–3 few-shot JSON examples.  
- Ban list: third-party IP names, NSFW, etc.

Affinity table is **structural inspiration** for gameplay-like character sheets only; all names and lore text are original.

### 7.3 L1 eval set

- 20–50 cases: normal briefs, short prompts, locked affinity, boundary/IP-injection attempts, optional ref cases.  
- Use 3 fixed cases for the demo video.  
- Lightweight checks: JSON parse, affinity ∈ set, ranges, non-empty `image_prompt`, banlist.

### 7.4 L2 LoRA (if any)

- Legal sources only (self-made / licensed / explicitly AI-trainable).  
- **Forbidden:** third-party show/movie frames, unauthorized scrapes.  
- 50–200 images, simple captions, optional trigger token.  
- Document license and before/after in PDF.

### 7.5 User reference images

- Inference-time only; not added to training by default.  
- Product copy: do not upload infringing character art; selfies used as pose/hair reference for original anime OC.

---

## 8. Repository structure

```
project/
  app.py
  configs/
    world.yaml
    models.yaml
  src/
    orchestrator.py
    llm_role.py
    image_gen.py
    video_gen.py
    card_compose.py
    device.py
  prompts/
    system_role.txt
    json_schema.txt
    few_shot.jsonl
    repair_json.txt
  data/
    eval/cases.jsonl
    lora/                 # optional, often gitignored bulk
  outputs/                # gitignore
  models/                 # gitignore
  scripts/
    setup_rocm.sh
    download_models.sh
  docs/
    samples/              # a few golden outputs for README/PDF
  requirements.txt
  README.md
```

---

## 9. Twelve-day plan

Anchor: design date 2026-07-26 → deadline 2026-08-06.

| Day | Focus | Exit criterion |
|-----|--------|----------------|
| D1 | Cloud access, repo skeleton, device check, Gradio shell + MOCK | App opens local + cloud |
| D2 | World.yaml, JSON contract, orchestrator states, output dirs | MOCK full path writes artifacts |
| D3 | Local LLM JSON + retry; start eval cases | ≥8/10 valid JSON |
| D4 | SDXL (or fallback) portrait on ROCm | Stable one portrait; OOM path exists |
| D5 | **Card compose + true E2E card** | **P0 done:** downloadable card from real models |
| D6 | Reference image path | With/without ref both work |
| D7 | Revise-from-result; VRAM profiles; ≥20 eval cases | Edit reflects on new card |
| D8 | I2V spike on cloud | One real video **or** freeze degrade plan |
| D9 | Video polish or stop-loss; optional LoRA only if spare | Main demo script never depends on video every run |
| D10 | Stability: 5 diverse cases; meta timings; UX errors | 5/5 cards; ≥3 distinct looks |
| D11 | UI polish; README; PDF draft; PR title | Cold start to card per README |
| D12 | Demo video; PDF final; PPT/poster; open PR | Complete submission package |

**Hard gate:** If no real card by end of D5, cut non-P0 (including video/polish) until card ships.

---

## 10. Definition of Done

### 10.1 P0 (blocking)

1. End-to-end character card on AMD Radeon + ROCm.  
2. Outputs: structured lore + portrait + affinity visualization + downloadable poster.  
3. Critical inference local (at least image gen on Radeon; LLM local for production path).  
4. Workbench web UI main path complete.  
5. Reproducible README; no secrets in repo.  
6. No third-party IP branding or infringing training data.

### 10.2 P1

1. Reference image influences portrait.  
2. Revise-from-result works.  
3. Live 2–4s ability video **or** honest degradation with P0 intact.

### 10.3 Bonus

- Small style LoRA with comparison.  
- `rc-tunnel` demo.  
- Eval regression script.  
- Documented 16/24GB profiles.

---

## 11. Submission package (Track 1)

Per official rules / [Radeon-hackathon-2026-07](https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07):

1. **Project Profile PDF (English):** background, users/scenarios, architecture, models/algorithms, AMD Radeon/ROCm adaptation.  
2. **Source code + README:** env, startup, dependencies, model download.  
3. **Demo video 3–5 min:** real AMD run; clarity, stability, diversity.  
4. **PPT or poster** (one required).  
5. **PR:** fork official repo; title like `Track 1, <Team>, <AppName>`; materials in English.

### 11.1 Demo video outline

| Time | Content |
|------|---------|
| 0:00–0:30 | Name, scenario, AMD |
| 0:30–0:50 | `rocm-smi` / device check |
| 0:50–1:10 | Start Web UI |
| 1:10–2:40 | Case 1 full card |
| 2:40–3:20 | Case 2 ref or revise |
| 3:20–4:20 | Video success or graceful limit + still diversity |
| 4:20–5:00 | One-slide architecture |

Record on AMD environment, not MOCK-only.

### 11.2 Eligibility reminders

- Register on Luma; join AMD Developer Program (China link if applicable).  
- Prize eligibility requires Developer Program membership.  
- Destroy idle cloud instances; keep PVC for weights if needed.

---

## 12. Risk register

| Risk | Mitigation |
|------|------------|
| ROCm image model issues | Swap checkpoint; SD1.5; lower res; known-good container |
| LLM JSON flaky | Stronger few-shot; 3B; repair pass; light field fill |
| I2V unsupported | AnimateDiff then drop live video |
| Cloud credits | MOCK locally; batch cloud sessions; destroy when idle |
| Scope creep (chat, big train, long video) | Reject; return to this spec |
| IP complaint | Original names only; no franchise assets |

---

## 13. Open items for implementation plan (not blockers)

These are intentionally deferred to implementation planning, not design ambiguity:

- Final commercial-friendly SDXL checkpoint ID and license text.  
- Exact I2V model ID after ROCm smoke test.  
- Final original world name and six affinity Chinese labels (structure fixed; copy editable).  
- Team name and English app name for PR title.  
- Whether Gradio custom CSS theme is minimal or slightly richer (both OK if P0 date holds).

---

## 14. Spec self-review (2026-07-26)

| Check | Result |
|-------|--------|
| Placeholders | No TBD for architecture; open items listed in §13 with owners = implementer |
| Contradictions | Card hard / video soft consistent across UI, arch, schedule |
| Scope | Single app, single GPU process, 12-day fit |
| Ambiguity | Production LLM path = local; platform free chat API = non-core only |
| Training | Explicitly optional; rules compliance explained |
| IP | Explicit ban on third-party franchise use |

---

## 15. Approval

- Design sections 1–5 discussed and confirmed with user in session (2026-07-26).  
- **Next after user approves this file:** invoke `writing-plans` to produce a step-by-step implementation plan.  
- **Do not start implementation coding until this spec is accepted.**
