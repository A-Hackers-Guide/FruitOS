# FruitOS

[![Download FruitOS](https://img.shields.io/sourceforge/dt/fruitos.svg?label=SourceForge%20downloads)](https://sourceforge.net/projects/fruitos/files/latest/download) [![Download FruitOS](https://a.fsdn.com/con/app/sf-download-button)](https://sourceforge.net/projects/fruitos/files/latest/download)


**A live Linux system whose star is a real fruit fly brain.** FruitOS boots straight into a 3D learning
app that runs the complete wiring diagram of a male fruit fly's central nervous system as a live simulation:
**165,122 neurons, 25.6 million connections, 124 million synapses.** The fly's brain is in control: what the
fly does on screen is decided only by its own command neurons. You can talk to it, poke its senses, and
take guided lessons.

Made for doctors, anthropologists, students and anyone curious about how brains work.

**Website:** https://a-hackers-guide.github.io/FruitOS/ (spin all 165,122 neurons in your browser)

- Based on **Arch Linux** (archiso), boots with **GRUB** (UEFI; legacy-BIOS PCs fall back to syslinux)
- Fully **offline**: no account, no internet, no AI service
- Built ISOs are published only as **[Releases](../../releases)**; this repository holds source code only

## What you can do

| | |
|---|---|
| **See the brain in 3D** | Every neuron drawn where its cell body sits, coloured by brain part. Spin, zoom, click any dot to inspect that neuron and its strongest partners. |
| **Watch it live** | The brain runs continuously. Neurons light up as they fire. |
| **The brain is in control** | Press *Shadow* and nothing is scripted: looming-detector neurons fire, drive the Giant Fiber, and the fly escapes. Behaviour comes only from command neurons proven in experiments (table below). |
| **Talk to the fly** | Ask "what are you doing?", "tell me about MDN", "what is GABA?", "what happens when you smell something?". Every answer is computed from the connectome or from a fresh simulation, never invented. |
| **Learn** | 9 guided lessons with quizzes whose answers are calculated from the brain you're looking at. |

### Behaviour comes from real command neurons

| Command neuron | What the fly does | Shown by |
|---|---|---|
| DNp01 (Giant Fiber) | escape jump / take-off | von Reyn et al. 2014 |
| MDN (moonwalker) | walk backward | Bidaye et al. 2014 |
| DNa01, DNa02 | steer left / right | Rayshubskiy et al. 2020; Yang et al. 2023 |
| DNp09 | walk forward | Bidaye et al. 2020 |

When the brain fires other command neurons, the fly says which ones, and says so when science doesn't yet
know what they do.

## Try it

**FruitOS ISO:** download the latest `fruitos-*.iso` from [Releases](../../releases), check it with
`sha256sum -c`, then boot it in a VM (VirtualBox, VMware, QEMU, UTM, with UEFI enabled) or write it to a USB stick.

On a real PC, turn **Secure Boot off** in the firmware settings first (FruitOS's boot files aren't signed for Secure Boot).

| | Minimum | Recommended |
|---|---|---|
| CPU | x86_64, 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Graphics | any (software rendering works) | any GPU with OpenGL |

**Just the app, on macOS / Linux / Windows (no ISO):**

```bash
pip install numpy scipy pyarrow pandas
bash tools/fetch_data.sh raw            # 1.1 GB from Janelia's public bucket
python3 tools/prepare_data.py --raw raw --out data
cd app && FRUITOS_DATA=../data python3 -m fruitbrain.server
```

Then open http://127.0.0.1:8642.

## How the brain is simulated (and its limits)

Each neuron is a leaky integrate-and-fire unit, following Shiu et al. 2024 (*Nature*, "A *Drosophila*
computational brain model reveals sensorimotor processing"). Each synapse adds a voltage kick whose sign
comes from the neuron's predicted transmitter: acetylcholine excites; GABA, glutamate and histamine inhibit.
For this larger dataset FruitOS changes three things, all stated in the app:

- **0.15 mV per synapse** (Shiu et al. used 0.275 mV on the FlyWire brain-only dataset; at that strength this
  whole-CNS dataset saturates).
- **Spike-frequency adaptation:** each spike raises that neuron's threshold by 4 mV, decaying over 100 ms.
- **Kenyon cell → Kenyon cell synapses (642,932) are left out of the simulation** (they're kept in the data).
  They are mostly axo-axonal and make point-neuron models run away.

Not modelled: neuromodulators (dopamine, serotonin, octopamine), gap junctions, neuron shapes, a body, and
learning. Sensory "senses" drive one real receptor cell type each (for example ORN_DA1 for smell; LPLC2 and
LC4 looming detectors for the shadow). Sight starts at the looming detectors because the photoreceptor →
lamina step works by release from inhibition, which this neuron model can't represent.

## Resource use

FruitOS is built to be light on the computer it runs on:

- The brain uses **at most half of one CPU core** (`FRUITOS_CPU_SHARE`), and systemd caps it at 1 core and 2 GB RAM.
- It **pauses when nobody is watching** (no app activity for 60 s).
- About 160 MB of brain data; about 0.2 to 1 GB of RAM while running. **No GPU required.**
- The fly lives in slow motion: about 1 second of brain time every 7 seconds on a typical laptop core.

## Repository layout

```
app/fruitbrain/   brain engine, live behaviour loop, the fly's voice, lessons, local web server
app/web/          3D learning interface (three.js, vendored, works offline)
tools/            download + convert the connectome (build time only)
iso/              archiso profile: packages, GRUB/syslinux branding, systemd service, kiosk session
tests/            engine tests on a tiny synthetic brain
site/             the GitHub Pages website
.github/workflows check.yml (tests) · release.yml (tag v* → ISO → Release) · pages.yml (website)
```

## Releasing

```bash
git tag v0.1.0 && git push origin v0.1.0
```

GitHub Actions builds the ISO inside an Arch Linux container: it downloads and converts the connectome, then
runs `mkarchiso`, and attaches the ISO plus a SHA-256 checksum to a Release. Nothing built is ever committed.

## Licences and credits

Code: MIT. Brain data: **CC BY 4.0**, MaleCNS v1.0 by HHMI Janelia (FlyEM), Google and collaborators,
https://male-cns.janelia.org. Full details, changes made to the data, and third-party licences:
[DATA_LICENSE.md](DATA_LICENSE.md). FruitOS is not affiliated with Arch Linux, Janelia or Google.
