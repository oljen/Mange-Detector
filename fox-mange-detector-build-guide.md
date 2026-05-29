# The Fox Mange Detector: A Complete Build Guide

*An end-to-end plan for building an AI system that spots sarcoptic mange in wild foxes early enough to save them — and gets that information into the hands of people who can treat them.*

---

## Why this matters (the thing to keep in mind the whole way through)

Sarcoptic mange is the single biggest disease killer of urban and suburban red foxes. It is slow, painful, and almost always fatal if left untreated — but it is also cheaply and reliably curable if caught early. The entire problem is **timing**: by the time a human happens to notice that a fox looks sick, the animal is often too far gone to save.

Your system's job is not to "cure foxes." It is to **close the detection gap** — to notice fur loss and skin lesions weeks before a casual human observer would, and to hand that information to a wildlife rehabilitator who can act. The technology is a smoke alarm, not a fire engine. Keeping that framing in mind will keep you from over-engineering and will keep the project genuinely useful.

The good news, confirmed by the current research literature: the hard infrastructure (edge AI cameras, detection models, alerting) already exists and is borrowable. The genuinely unsolved piece — the thing where you can make a real contribution — is a usable, labeled dataset of mangy vs. healthy foxes and a field-deployed pipeline built around it.

---

## How the finished system works (the pipeline)

Picture the data flowing left to right:

```
[ Camera in the field ]
        |  motion/heat trigger captures an image
        v
[ MegaDetector ]  -> finds the animal, crops it out, discards empty frames
        |  "there is an animal here, at these coordinates"
        v
[ Species filter ]  -> is this actually a fox? (ignore cats, raccoons, people)
        |  "yes, red fox"
        v
[ Mange classifier ]  -> YOUR model: healthy / suspected-mange + confidence
        |  "suspected mange, 0.87 confidence"
        v
[ Alert logic ]  -> only fires when confident + not a duplicate of last alert
        |
        v
[ Wildlife rehabber ]  -> gets a phone notification with the photo + location
        |
        v
[ Treatment ]  -> rehabber confirms, administers medication, fox recovers
```

Two design principles run through the whole thing:

1. **It is passive and non-habituating.** The fox never associates the device with food or humans. It just watches. This is an ethical and ecological requirement, not a nice-to-have.
2. **A human is always the final decision-maker.** The model flags; a person confirms and acts. This protects against false positives harming the animal (e.g. an unnecessary trapping attempt) and keeps you on the right side of wildlife law.

---

## The five phases

You will build this in five phases, roughly in order, though Phase 0 runs in parallel with everything.

- **Phase 0 — Foundations:** partnerships, ethics, and the dataset problem
- **Phase 1 — Proof of concept:** a crude classifier on your laptop (one day)
- **Phase 2 — The real dataset:** the hard, important part
- **Phase 3 — The full software pipeline:** detection + species + mange, chained together
- **Phase 4 — Hardware & field deployment:** putting it outside
- **Phase 5 — Closing the loop:** alerts, rehabber workflow, and iteration

---

## Phase 0 — Foundations (start this on day one and never stop)

### 0.1 Find a wildlife rehab / research partner

This is the highest-leverage thing you will do, and it has a long lead time, so start immediately. You need a partner for three reasons:

- **Data.** Rehabbers and university ecology groups have photo archives of mange cases — exactly the labeled images you cannot get anywhere else. The research community's biggest bottleneck is that no public, ML-ready fox-mange dataset exists. Your partner is how you solve that.
- **Ground truth.** They can confirm whether a flagged fox actually has mange, which is how you measure if your model works.
- **Action.** They are the ones who can legally and humanely treat a flagged fox. Without them, your system detects suffering it cannot relieve.

How to start: write one short, warm email today to a local wildlife rehabilitator or a nearby university's ecology/veterinary department. Explain plainly what you're building and that you'd value their guidance and, eventually, any anonymized mange photos they'd be willing to share. In the US you can find rehabbers via your state wildlife agency or directories like wildliferehab.org.

### 0.2 Understand the legal and ethical guardrails

Before you point a camera at wildlife, know the rules where you live:

- **Camera placement & privacy.** Cameras that might capture people (especially near homes or that demolished complex) raise privacy and sometimes legal issues. Aim cameras at den/trail areas, away from where people pass, and check local surveillance laws.
- **Wildlife handling.** You almost certainly may *not* trap, touch, or medicate a wild fox yourself. That is what your licensed rehab partner is for. Your role ends at "flag and notify."
- **Land permission.** Get permission from whoever owns or manages the site.
- **Data on at-risk animals.** Don't publish precise den locations publicly; it can attract harassment or poaching.

### 0.3 Set up your tools

You'll want, for free:
- A **Google account** for Google Colab (free in-browser notebooks with GPU access — no powerful computer required to train).
- A **GitHub account** to version your code and pull open-source tools.
- Basic **Python** comfort. If you're rusty, that's fine — the heavy lifting is done by libraries, and you can learn as you go.

---

## Phase 1 — Proof of concept (a single day on your laptop)

The goal of this phase is to answer one question cheaply: *can a model tell a mangy fox from a healthy one at all?* If yes, the project is worth your time and money. No hardware needed.

### 1.1 Gather a tiny starter dataset

Use **iNaturalist** — a citizen-science platform with thousands of red fox observations, many openly (Creative Commons) licensed and some tagged with mange. Pull roughly:
- 75–100 photos of visibly mangy red foxes (search observations, filter for mange, check each license)
- 75–100 photos of healthy red foxes

Put them in two folders: `healthy/` and `mange/`. This is tiny and imperfect — that's fine. Today is a feasibility test, not a finished model.

> **License discipline starts now.** Record the license and source of every image in a spreadsheet. You'll thank yourself later when you want to publish or share the dataset.

### 1.2 Crop to the animal with MegaDetector

**MegaDetector** (Microsoft's free, open-source AI for Good tool) finds animals in wildlife photos and draws bounding boxes around them. It is *a detector, not a classifier* — it tells you "there's an animal here," not what kind or whether it's sick. You use it to crop each photo down to just the fox, so your downstream model focuses on the animal and not the background.

The current version is **MegaDetector V6**, accessible through the **PyTorch-Wildlife** framework, which downloads the model weights automatically. A few lines of Python in Colab:

```python
from PytorchWildlife.models import detection as pw_detection
model = pw_detection.MegaDetectorV6()
results = model.single_image_detection("path/to/fox.jpg")
# use the returned bounding box to crop the image to the animal
```

### 1.3 Fine-tune a pretrained classifier (transfer learning)

Don't train from scratch — you don't have the data and don't need to. Take a model that already understands images (a **ResNet** or **EfficientNet** from PyTorch or TensorFlow, pretrained on ImageNet) and fine-tune it on your two classes. You're borrowing a model that already understands fur, texture, and edges, and just teaching it your specific distinction.

In Colab with a free GPU this is an afternoon's work even for a relative beginner. The pattern:
1. Load the pretrained model, replace its final layer with a 2-class output (healthy / mange).
2. Freeze most layers, train only the last few on your cropped fox images.
3. Hold back ~20% of your images as a test set the model never sees during training.

### 1.4 Look at the mistakes (this is the real insight)

Get an accuracy number, but spend your real attention on the images the model got *wrong*. Mange shows as patchy fur loss, crusty skin, and a thin scraggly tail. If the model latches onto those features — great. If it's fooled by wet fur, shadows, or fox cubs that naturally look patchy, you've just discovered exactly what your real dataset must cover.

**Decision gate:** If even this crude model gets meaningfully above chance (say 75%+ on your held-out test set), the core idea is sound. Proceed. If it's hopeless, you've learned that for about $0 and a day.

> **Reality check:** A model trained on ~200 clean web photos is a feasibility probe, not a diagnostic tool. Do not trust its judgment on a real fox yet. And clean daytime iNaturalist photos are very different from grainy 3 a.m. infrared camera-trap frames — bridging that gap is Phase 2's job.

---

## Phase 2 — Build the real dataset (the hard, important part)

This is where the project is won or lost, and where your contribution to the world actually lives. The research is unanimous: *the barrier is data, not model architecture.*

### 2.1 What "good data" means here

You need images that look like what your field camera will actually see:
- **Real camera-trap conditions:** infrared/night shots, motion blur, partial bodies, bad angles, rain, varying distances.
- **A range of mange severity:** early/subtle cases matter most, because early detection is the whole point. A dataset of only late-stage, obvious mange teaches the model to catch foxes too late to save.
- **Hard negatives:** healthy foxes that *look* a bit off (wet, molting, cubs), plus other animals (cats, raccoons, dogs) so the model learns "not a fox" and "fox but healthy."
- **Geographic and seasonal variety** if you want it to generalize beyond your one site.

### 2.2 Where the data comes from (stack these sources)

1. **Your rehab partner's archives** — the gold. Treated mange cases, often with confirmed diagnoses.
2. **Your own cameras** — once Phase 4 is running, every image your system captures becomes training data (this is the flywheel: deployment improves the model).
3. **iNaturalist / GBIF / public citizen science** — for volume and variety, with license tracking.
4. **Other researchers** — ask. Several groups have labeled mange images in foxes, coyotes, and wolves; collaboration is common in conservation tech.

### 2.3 The synthetic-data trick (if real data is scarce)

If you genuinely can't get enough real early-mange images, recent work generates *synthetic* training images — taking healthy camera-trap photos and editing realistic hair-loss/emaciation onto them to expand the dataset. This is an advanced, optional technique, but worth knowing it exists, because it's exactly the approach researchers are using to get around the same scarcity you'll hit.

### 2.4 Label carefully and conservatively

- Two classes to start: `healthy` and `suspected_mange`. (You can add severity grades later.)
- Have your rehab partner validate a sample of your labels. "Mange-compatible lesions" judged from a photo are not a confirmed clinical diagnosis — be honest about that distinction in how you describe the system.
- Track every image's source, license, date, and label in a structured file.

### 2.5 Aim points

There's no magic number, but as rough orientation: a few hundred images per class can give a usable first model; a few thousand well-varied, well-labeled images is where field reliability starts to become realistic. Quality and variety beat raw count.

---

## Phase 3 — The full software pipeline

Now you chain the pieces into the flow shown earlier. A proven open-source template exists: a University of Cambridge project ("AI for Wildlife Monitoring") runs essentially this architecture — a cellular camera trap sends photos to a server or Raspberry Pi, MegaDetector plus a species classifier process them, and alerts go to a phone via Telegram. You can study and adapt it rather than starting from a blank page.

### 3.1 Stage one: MegaDetector finds the animal

Every incoming frame goes through MegaDetector first. It discards empty frames (wind, swaying branches — the majority of triggers) and crops out any animal. This alone removes the single biggest workload in camera-trap systems.

### 3.2 Stage two: is it a fox?

Run the cropped animal through a **species classifier** so you only analyze foxes and ignore cats, raccoons, deer, and people. You can use an existing open-source camera-trap species classifier (several cover North American and European mammals) rather than building your own.

### 3.3 Stage three: your mange classifier

The fox crop goes to *your* model from Phases 1–2, which outputs `healthy` or `suspected_mange` plus a confidence score.

### 3.4 Stage four: smart alert logic

Don't alert on every frame. Add rules so the system is useful rather than annoying:
- **Confidence threshold:** only flag above, say, 0.8 (tune this with your partner — false negatives and false positives have very different costs here).
- **Temporal smoothing:** require the same flag across several frames or sightings before alerting, to filter out one-off glitches.
- **De-duplication:** don't re-alert on the same fox every night for a week. Track recent alerts and suppress repeats.
- **Quiet confirmation packet:** when it does fire, send the cropped image, the full-frame context, the confidence score, the timestamp, and a rough location.

---

## Phase 4 — Hardware & field deployment

You have two architectures. Choose based on your site.

### Option A — "Smart camera at the edge" (everything runs on-site)

Best when you have power/solar and want a self-contained unit.

| Component | Recommended | Approx. cost (USD) |
|---|---|---|
| Compute | Raspberry Pi 5 (8GB) | ~$80 |
| AI accelerator | Raspberry Pi AI HAT+ (Hailo-8L, 13 TOPS) | ~$70 |
| Camera | Raspberry Pi Camera Module 3 (or a NoIR version for night) | ~$25 |
| Case + power/solar + storage | weatherproof case, solar panel + battery, microSD | ~$60–150 |
| **Base electronics subtotal** | | **~$195 + power** |

The **AI HAT+** matters because it runs the neural-network inference on a dedicated low-power chip (the Hailo NPU), entirely on-device — no cloud, no subscription, low power draw, and it's natively integrated into the Pi's camera software stack. It's the same class of hardware recent research prototypes use for real-time on-device wildlife detection. (There's also a newer **AI HAT+ 2** at ~$130 with more capability, and **Google Coral** as an alternative accelerator — the Hailo is the better-supported default in 2026.)

For night vision, use a **NoIR camera + infrared illuminator** so you can see nocturnal foxes without disturbing them with visible light.

### Option B — "Cellular camera + remote brain" (simplest to start)

Best when the site has no reliable power for a full compute unit. Buy an off-the-shelf **4G/LTE cellular trail camera** that emails or uploads its photos. Those photos land on a cheap home server, spare PC, or even a Raspberry Pi 4B elsewhere, which runs the pipeline and sends alerts. This is exactly the Cambridge template's approach and is the fastest path to a working field system. Downside: ongoing data/SIM cost and dependence on cell coverage.

### 4.1 Connectivity for alerts
- **Cellular** (SIM in the camera or a cellular hat) where there's coverage.
- **Wi-Fi** if the site is near a building (like the edge of your complex).
- **LoRa / satellite** for truly remote sites (satellite real-time alerting has been demonstrated for wildlife, but it's more advanced and costly).

### 4.2 Power
Solar panel + rechargeable battery is standard for unattended deployment. Size the panel to your compute draw — the Pi + AI HAT pulls more than a bare trail camera, so Option B is gentler on power.

### 4.3 Field-hardening
- Weatherproof (IP-rated) enclosure with proper cable glands.
- Mount where it sees the fox's regular trails/den approach, angled to avoid capturing people.
- Camouflage and secure it against theft and against the foxes themselves investigating it.
- Test the whole chain indoors first, then in your yard, then at the site.

---

## Phase 5 — Closing the loop (the part that actually saves foxes)

A flag that nobody acts on saves no animals. Design the human workflow as carefully as the software.

- **Delivery:** send alerts to your rehab partner the way *they* will actually see them — often a phone notification via a messaging app (Telegram bots are a common, free, easy choice for this).
- **Make confirmation one tap:** let the rehabber mark each alert "confirmed mange," "healthy — false alarm," or "can't tell." This both triggers action *and* feeds labels back into your dataset.
- **The flywheel:** every confirmation makes your next model better. Retrain periodically with the new field-labeled data. Over a season, your site-specific accuracy should climb.
- **Track outcomes:** how many foxes flagged, how many confirmed, how many treated, how many recovered. This is both your impact metric and what you'll show future partners and funders.

---

## How this actually helps the world

Be honest and specific about the impact, because that's what makes it real:

- **Per fox:** mange caught early is a cheap, routine cure instead of a slow death. Each true early detection that leads to treatment is, plausibly, one fox's life.
- **Per family/territory:** mange spreads within fox families and between neighboring foxes. Catching one early can prevent an outbreak, protecting kits like the one you saw.
- **Per community:** the same mite affects dogs and other wildlife. Early detection in the local fox population reduces spillover risk.
- **For science:** a labeled fox-mange dataset and a working field pipeline are things the research community explicitly says don't yet exist. If you open-source them (with your partner's consent and proper licensing), you don't just help your foxes — you give every other rehabber and researcher a head start. *That* is how a one-person project scales beyond one backyard.

The scaling path: one unit at your site → a handful across a city in partnership with a rehab network → an open dataset and open-source design others replicate → mange detection becomes a standard, cheap tool in urban wildlife care. None of that requires you to do it all; it requires you to build the first working node and share what you learn.

---

## Bill of materials (starter, Option A)

| Item | Approx. USD |
|---|---|
| Raspberry Pi 5 (8GB) | 80 |
| Raspberry Pi AI HAT+ (Hailo-8L) | 70 |
| Camera Module 3 (NoIR for night) | 25–35 |
| IR illuminator | 15–30 |
| Weatherproof case | 20–40 |
| Solar panel + battery | 40–100 |
| microSD + power bits | 20 |
| **Total** | **~$270–375** |

Software is essentially free: Colab (free tier), MegaDetector, PyTorch-Wildlife, open species classifiers, Python, and a Telegram bot all cost nothing. Your real "cost" is time and the dataset effort.

---

## Common pitfalls (learn these now, not the hard way)

- **Skipping the partner.** Without a rehabber, you build a system that detects suffering it can't relieve, and you can't get ground-truth data. Start that relationship first.
- **Training only on obvious, late-stage mange.** You'll build a detector that catches foxes too late to save. Prioritize subtle, early cases.
- **Trusting the model too early.** A few hundred web images is a prototype, not a diagnosis. Always keep a human in the loop.
- **Ignoring the day/night domain gap.** Models trained on daytime photos often fail on infrared night frames. Include real night data.
- **Habituating the foxes.** No food, no attractants, no bright lights. The device must be invisible to the fox's behavior.
- **Privacy and permissions.** Cameras that catch people, or placement without landowner permission, can cause real legal trouble. Sort this before deploying.
- **Alert fatigue.** A system that cries wolf gets ignored. Tune thresholds and de-duplication with your partner.

---

## A realistic timeline

- **Day 1:** Phase 1 proof of concept + send the partnership email.
- **Weeks 1–4:** secure a partner, settle ethics/permissions, start gathering real data.
- **Months 1–3:** build the real dataset; get the full software pipeline running on a laptop/server with Option B (cellular camera).
- **Months 3–6:** field-deploy your first unit, start collecting real-world data and your first true alerts.
- **Months 6–12:** iterate with field-labeled data, improve accuracy, document results, consider a second site and open-sourcing.

You don't need to finish all of this to help a fox. A working Option-B setup at your site, feeding confirmed alerts to a local rehabber, is already saving animals long before the "finished" version exists.

---

## First three concrete actions

1. **Today:** open Colab, pull ~150 fox images from iNaturalist, run the Phase 1 proof of concept, and look at what the model gets wrong.
2. **Today:** send one warm email to a local wildlife rehabber or university ecology dept introducing the project and asking for guidance + any mange photos.
3. **This week:** read and clone the Cambridge "AI for Wildlife Monitoring" camera-trap repo and the MegaDetector / PyTorch-Wildlife docs so you understand the proven template you're adapting.

Build the smoke alarm. Let the rehabbers be the fire engine. The foxes get the rest.
