# `gammachirp-rs` Rust crate

Dynamic compressive gammachirp filterbanks, provided as a Rust port of the
original GammachirPy Python implementation.

## Rust crate

The Rust port provides `gcfb_v234::{gammachirp, utils, gcfb_v234}` for frame
processing, audiograms, cochlear hearing loss, synthesis, and
envelope-modulation tools.

Filterbank matrices use the same channel-major orientation as Python: rows are
channels and columns are samples or frames. Parameters are typed structs, and
invalid inputs return `gammachirp_rs::Result`.

```rust
use gammachirp_rs::gcfb_v234::{GcParam, gcfb_v234};

let input = [1.0, 0.0, 0.0, 0.0];
let parameters = GcParam {
    num_ch: 32,
    out_mid_crct: "No".into(),
    ..GcParam::default()
};

let output = gcfb_v234(&input, parameters)?;
assert_eq!(output.dcgc_out.nrows(), 32);
# Ok::<(), gammachirp_rs::Error>(())
```

### Sample streaming

The v2.34 bounded-memory `GcfbStream` always returns the immediate sample-domain
`scgc_smpl`.
Static, level, and dynamic sample-base modes also return a `DcgcEvent::Sample`
immediately. Dynamic frame mode delays `DcgcEvent::Frame` until the centered
window's right-hand samples are available. After the input ends, `finish()` is
required to emit the remaining zero-padded frames. For `N` samples it brings
the complete event count to `N / len_shift + 1`, matching the batch API.

```rust
use gammachirp_rs::gcfb_v234::{
    ControlMode, DcgcEvent, DynHpaf, GcParam, GcfbStream,
};

let mut filterbank = GcfbStream::new(GcParam {
    num_ch: 32,
    out_mid_crct: "No".into(),
    ctrl: ControlMode::Dynamic,
    dyn_hpaf: DynHpaf {
        str_prc: "frame-base".into(),
        ..DynHpaf::default()
    },
    ..GcParam::default()
})?;

let mut frames = Vec::new();
for sample in [1.0, 0.0, 0.0, 0.0] {
    if let Some(event) = filterbank.process_sample(sample)?.event {
        frames.push(event);
    }
}
frames.extend(filterbank.finish()?);
assert!(frames.iter().all(|event| matches!(event, DcgcEvent::Frame { .. })));
# Ok::<(), gammachirp_rs::Error>(())
```

Every emitted `dcgc_out` already includes the selected dcGC gain
normalization. The parameter and response accessors expose prepared,
time-invariant metadata such as `gain_factor` and `cgc_ref`; time-varying
histories remain attached to their sample or frame events. Streaming uses
direct causal FIR convolution while the optimized batch path uses FFT
convolution, so their finite outputs agree to ordinary floating-point
roundoff rather than being guaranteed bit-for-bit identical.

### Breebaart 2001 binaural processing

The `breebaart2001` module combines the dynamic compressive gammachirp
filterbank with the inner-hair-cell and adaptation-loop stages and the
contralateral-inhibition stage described by
[Breebaart, van de Par, and Kohlrausch (2001)](https://doi.org/10.1121/1.1383297).
This is intentionally a hybrid: the original paper used a linear gammatone
filterbank, whereas `hybrid_binaural` uses this crate's GCFB v2.34. Call
`breebaart2001_ei` directly to apply the paper's EI equations to an existing
peripheral representation in model units.

Peripheral arrays have shape `(frequency channel, sample)`. EI maps add the
population axis and have shape `(unit, frequency channel, sample)`. The
pre-inner-hair-cell absolute-threshold noise and the post-EI internal noise are
separate, reproducibly seeded sources; disable both for deterministic,
noise-free activity maps.

```rust
use gammachirp_rs::breebaart2001::{
    EiConfig, EiUnit, breebaart2001_ei,
};
use ndarray::Array2;

let left = Array2::from_elem((32, 256), 10.0);
let right = left.clone();
let units = [
    EiUnit::default(),
    EiUnit::new(0.0, 3.0),
    EiUnit::new(0.5e-3, 0.0),
];
let config = EiConfig {
    internal_noise_std_mu: 0.0,
    ..EiConfig::default()
};

let activity = breebaart2001_ei(&left, &right, 48_000.0, &units, &config)?;
assert_eq!(activity.dim(), (3, 32, 256));
# Ok::<(), gammachirp_rs::Error>(())
```

#### Breebaart sample streaming

The monaural, EI-only, and end-to-end hybrid stages also have bounded-memory
`MonauralStream`, `EiStream`, and `HybridBinauralStream` processors. The
paper's double-sided exponential and AMT's forward-backward EI filter are
acausal, so streaming is selected explicitly with `MonauralConfig::streaming()`,
`EiConfig::streaming()`, or `HybridBinauralConfig::streaming()`. Existing
defaults remain unchanged for offline work.

Each accepted waveform pair immediately returns GCFB and peripheral channel
vectors. EI activity is an optional indexed event because paper-symmetric
fractional delays require a fixed amount of future input. For a finite signal,
`finish()` emits that bounded zero-extended tail. For an indefinite signal,
continue calling `process_sample()` and do not call `finish()`.

```rust
use gammachirp_rs::breebaart2001::{
    EiUnit, HybridBinauralConfig, HybridBinauralStream,
};

let mut config = HybridBinauralConfig::streaming();
config.filterbank.num_ch = 16;
config.filterbank.out_mid_crct = "No".into();
let units = [EiUnit::default(), EiUnit::new(0.5e-3, 3.0)];
let mut processor = HybridBinauralStream::new(&units, config)?;

let mut ei_samples = Vec::new();
for (left, right) in [(1.0, 0.8), (0.0, 0.0), (0.0, 0.0)] {
    let step = processor.process_sample(left, right)?;
    if let Some(event) = step.ei_event {
        ei_samples.push(event);
    }
}
ei_samples.extend(processor.finish()?);
assert_eq!(ei_samples.len(), 3);
# Ok::<(), gammachirp_rs::Error>(())
```

Both seeded noise sources are traversed in sample-major order so noisy batch
and streaming calls reproduce the same values. This changes the exact seeded
sequence produced by earlier batch-only code while preserving deterministic
replay, trial-seed derivation, and the model's sharing/independence rules.

`hybrid_binaural` accepts equal-length left and right waveforms. It runs each
ear through the same configured GCFB and peripheral stages, then evaluates the
requested EI population. Dynamic GCFB configurations are evaluated in sample
mode to retain fine structure. Its output includes the EI map, both adapted
internal representations, center frequencies, and both complete GCFB outputs.

The runnable hybrid example generates deterministic broadband input at 16 kHz,
delays the right ear by 0.5 ms, makes it 3 dB louder than the left ear, and
analyzes it sample by sample with a noise-free 24-channel static GCFB and a
causal ITD-by-IID EI population:

```text
cargo run --example breebaart2001_hybrid
cargo run --example breebaart2001_hybrid -- /tmp/breebaart.png
```

It accumulates the requested interior-window means directly from indexed
stream events, flushes the delayed EI tail, and prints the stream latency along
with the stimulus, expected EI sign convention, dimensions, frequency range,
observed best-matching unit, and five lowest mean responses. The heatmap is
written to `target/breebaart2001_hybrid.png` by
default; one optional `.png` path selects another destination, and missing
parent directories are created automatically. In an EI population, a unit
whose characteristic ITD and IID match the stimulus cues cancels excitation
against inhibition. Cue matching therefore appears as a minimum in activity,
not a peak. With the paper's symmetric-delay convention, a right-ear waveform
delay of +0.5 ms is matched by a characteristic ITD of -0.5 ms and an IID of
+3 dB (defined here as the right-ear level minus the left-ear level).

The default `EiConfig` follows the continuous-time equations in the paper.
`EiConfig::amt_1_6()` instead selects AMT 1.6's one-sided integer delay,
2.2 ms delay weighting, forward-backward filter boundaries, and noise-free
EI-cell output. It also disables the paper's 5 ms and 10 dB population limits,
which the AMT EI cell does not enforce. AMT adds its internal noise in the later
central processor, so that decision noise remains the caller's responsibility.
This option matches the AMT EI cell; the end-to-end hybrid still uses the GCFB
rather than AMT's linear gammatone filterbank.

`CentralTemplate::fit` estimates the masker mean and variance and the expected
target-minus-masker difference from labeled trials. `score` applies the
Appendix-B weighting, and `choose_interval` selects the largest score in a
forced-choice task. For the paper's detector, select the single EI unit suited
to the experiment; do not pass the complete EI population, whose internal
noise is shared across units. The detector representation has axes `(detector
channel, frequency channel, sample)`, with the selected binaural channel and
any processed left/right monaural channels on its first axis.

The hybrid's `left_internal` and `right_internal` fields are raw adaptation-loop
outputs, not detector-ready monaural channels. Process them with
`breebaart2001_monaural`, which defaults to the paper's double-sided 10 ms
exponential and 0.0003 sensitivity. `MonauralConfig::amt_1_6()` selects AMT's
causal one-pole convention instead. The helper is deterministic; internal
noise required by the selected central decision model must be represented in
the trials or added at that later stage.

```rust,no_run
use gammachirp_rs::breebaart2001::{
    MonauralConfig, breebaart2001_monaural,
};
use ndarray::{Axis, stack};

# let output: gammachirp_rs::breebaart2001::HybridBinauralOutput = todo!();
let sample_rate = output.left_filterbank.gc_param.fs;
let monaural = MonauralConfig::default();
let left = breebaart2001_monaural(&output.left_internal, sample_rate, &monaural)?;
let right = breebaart2001_monaural(&output.right_internal, sample_rate, &monaural)?;
let selected_ei = output.ei_map.index_axis(Axis(0), 0);
let representation = stack(Axis(0), &[selected_ei, left.view(), right.view()])?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

When generating those training trials, derive each presentation from a common
base configuration with `config.for_trial(trial_index)`. This preserves exact
replay for a given index while giving every trial distinct absolute-threshold
and post-EI internal-noise realizations. Reusing an unchanged seeded
configuration intentionally replays the same noise and must not be used to
estimate trial-to-trial masker variance.

### Reassigned dcGC analysis

GCFB v2.34 also exposes an analysis-only, energy-conserving reassignment map.
On a shared finite DFT domain, it passes the original input and its exact time-
and DFT-derivative-weighted forms through the configured outer/middle-ear
correction FIR. It then projects each zero-padded real pGC onto the
nonnegative bins, applies the same HP-AF operator to all three inputs, and
computes coordinates without phase unwrapping. It bilinearly transports
analytic-representation power (`|C|^2 / 2`) onto the frame-time/ERB grid. This
is not pointwise or framewise `dcgc_out^2`, and the ordinary `GcfbOutput` is not
changed.

```rust
use gammachirp_rs::gcfb_v234::{GcParam, gcfb_v234_with_reassignment};

let input = [1.0, 0.0, 0.0, 0.0];
let (filterbank, reassigned) =
    gcfb_v234_with_reassignment(&input, GcParam {
        num_ch: 32,
        out_mid_crct: "No".into(),
        ..GcParam::default()
    })?;

assert_eq!(filterbank.dcgc_out.nrows(), 32);
assert_eq!(reassigned.energy_map.nrows(), 32);
assert!((reassigned.retained_energy() + reassigned.discarded_energy
    - reassigned.source_energy).abs() < 1e-10);
# Ok::<(), gammachirp_rs::Error>(())
```

The imaginary analysis branch is offline and acausal; the ordinary real GCFB
branch remains causal and unchanged. In frame mode, dynamic compression is a
positive real gain, so it affects transported energy but not coordinates. In
sample mode, all three operator applications replay the recorded HP-AF
coefficient history. The frequency coordinate then differentiates the realized
complex coefficient history itself, so it includes the HP-AF's temporal
variation without taking a derivative of the nonlinear estimator as a mapping
from input to coefficients. Fixed and frame modes are exact for their
implemented conditioned linear operator; sample mode is conditional on the
recorded history and exact for its zero-padded finite-DFT interpolant up to
floating-point error. The selected DFT length is exposed as
`ReassignmentResult::analysis_fft_len`.

For indefinite inputs, `ReassignmentStream` provides a bounded-memory,
zero-latency causal approximation. It owns the ordinary v2.34 stream and
returns that filterbank step together with per-channel source energy,
reassigned coordinates, and phase-transported complex contributions. Static,
level, and dynamic sample-base control are supported; dynamic frame-base
control remains batch-only.

```rust
use gammachirp_rs::gcfb_v234::{
    ControlMode, DynHpaf, GcParam, ReassignmentStream,
};

let mut analysis = ReassignmentStream::new(GcParam {
    num_ch: 32,
    out_mid_crct: "No".into(),
    ctrl: ControlMode::Dynamic,
    dyn_hpaf: DynHpaf {
        str_prc: "sample-base".into(),
        ..DynHpaf::default()
    },
    ..GcParam::default()
})?;

for sample in [1.0, 0.0, 0.0, 0.0] {
    let step = analysis.process_sample(sample)?;
    assert_eq!(step.source_energy.len(), 32);
    assert_eq!(step.phase_contribution.len(), 32);
    assert_eq!(step.filterbank.sample_index + 1, analysis.samples_processed());
}
assert_eq!(analysis.latency_samples(), 0);
# Ok::<(), gammachirp_rs::Error>(())
```

The stream uses a cosine-plus-sine causal gammachirp atom instead of the batch
path's whole-signal DFT projection. Fixed filters use its causal derivative;
sample-dynamic filters use the wrapped backward phase increment of the realized
complex coefficient, which includes changes in the HP-AF. Consequently, the
first nonzero dynamic coefficient has no frequency coordinate. Its real branch
follows the ordinary GCFB, while complex coordinates and phase are expected to
be similar rather than identical to batch reassignment away from finite-signal
boundaries. No relative coefficient floor is applied online:
`coordinate_mask` only reports nonzero coefficients with finite coordinates,
and callers can threshold `source_energy` for their own finite or rolling
target maps. The stream emits immediately and therefore has no `finish()`
tail.

`BandwidthConsensusStream` extends that causal analysis to an indefinite input
without retaining the signal. It runs one reassignment stream for every
bandwidth scale and keeps a bounded rolling target map. Once its normalization
window is full, each input finalizes one oldest agreement, mask, and salience
column. The individual scale steps remain immediate; consensus latency is one
less than the resolved window length. `scale_metadata()` reports each scale's
retuned passive carriers and measured continuous-DTFT nominal composite peaks.

```rust
use gammachirp_rs::gcfb_v234::{
    BandwidthConsensusStream, BandwidthConsensusStreamConfig, ControlMode,
    GcParam,
};

let mut consensus = BandwidthConsensusStream::new(
    GcParam {
        num_ch: 32,
        out_mid_crct: "No".into(),
        ctrl: ControlMode::Static,
        ..GcParam::default()
    },
    BandwidthConsensusStreamConfig {
        // None derives the horizon from the longest configured causal atom.
        window_samples: None,
        ..BandwidthConsensusStreamConfig::default()
    },
)?;

for sample in [1.0, 0.0, 0.0, 0.0] {
    let step = consensus.process_sample(sample)?;
    assert_eq!(step.scale_steps.len(), 3);
    if let Some(frame) = step.consensus {
        assert_eq!(frame.salience.len(), 32);
    }
}
// Finite callers flush delayed target columns; indefinite callers keep going.
let tail = consensus.finish()?;
# Ok::<(), gammachirp_rs::Error>(())
```

The streaming configuration deliberately omits `ReassignmentConfig`'s
coefficient floor because that threshold depends on a future whole-input
channel maximum. Each rolling scale is instead normalized by its maximum over
the active target window. The default window adapts to the prepared sample rate
and bandwidths; callers can set `window_samples` to a nonzero value for a fixed
horizon. Coordinates outside the live rolling time range or common ERB grid
are discarded. Memory is proportional to the fixed scales, channels, atom
support, and window size rather than elapsed input length.

Energy rejected by the relative coefficient floor and by target-map boundaries
is reported separately. Reassignment sharpens resolved ridges but cannot split
closely spaced components that already occupy the same gammachirp passband.

Phase-aware reassignment is available through
`gcfb_v234_with_phase_reassignment` and
`phase_reassign_gcfb_v234_with_config`. It follows the complex phase transport
proposed by [Gardner and Magnasco (2006)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1431718/).
For a source coefficient in channel `k`, it scales the analytic coefficient
so its squared magnitude is the transported energy, then applies

\[
\exp\!\left(i\pi(f_{r1,k}+\hat f)(\hat t-t)\right)
\]

Here `f_{r1,k}` is the analysis filter's actual passive carrier; for a scaled
consensus analysis this is the retuned carrier reported by `scale_metadata`,
while deposition still uses the shared target-frequency grid. After this
rotation and bilinear deposition, `PhaseReassignmentResult::complex_map` retains
absolute phase for analysis, while `phase_coherence_map` is the magnitude of
the complex sum divided by the sum of contribution magnitudes. The complex map
is a linearly interpolated amplitude histogram, not an energy map: a single
contribution deposited with weight `w` has complex-map power proportional to
`w^2`, while `energy_map` receives energy proportional to `w`. Multiple
contributions may additionally interfere. Empty bins have zero coherence. The
matched `unreassigned_energy_map` contains only the same floor- and
boundary-accepted contributions on the reassigned map's grid. Its time
coordinate is the source sample for sample-based analyses and the originating
frame for frame-based analyses.

`SparsityMetrics` reports Shannon entropy in nats, its exponential as the
effective number of bins, and the effective-bin fraction. A
`SparsityComparison` rejects maps whose retained energy differs, preventing
coefficient floors or boundary losses from making reassignment appear
artificially sparse. Deterministic white-noise tests in this crate show reduced
effective support for the matched GCFB maps; that is an empirical observation
of this implementation, not the paper's Gaussian-STFT white-noise theorem.

`gcfb_v234_with_bandwidth_consensus` runs phase-aware analysis at several
complete-filter bandwidths. Its default scales are `0.8`, `1.0`, and `1.2`.
Each scale multiplies both coefficients of `b1`, every coefficient of `b2`, and
`lvl_est.b2`; chirp, compression, level-control, hearing-loss, and target-grid
parameters remain fixed. Each scaled analysis retunes its internal passive
carriers so the implemented cosine-pGC FIR plus four-section digital HP-AF
cascade has the same continuous-DTFT main-lobe peak frequency as the baseline.
An internal FFT of at least 65,536 points locates the intended lobe nearest the
analytic or previously tracked peak; compensated arbitrary-frequency DTFT sums
and analytic HP-AF derivatives then bracket and bisect its continuous maximum.
The FFT is only a bracketing aid, and `scale_metadata` reports the resulting
continuous peak frequencies rather than grid information. If a maximum or
matching root cannot be bracketed, or the final frequency residual exceeds
`4096 * f64::EPSILON * max(sample_rate, 1)` Hz, preparation or processing
returns an error instead of selecting a nearby FFT bin.

In sample-dynamic mode, every scale keeps its own level history. At each
coefficient update, a scaled filter solves for the peak of the unscaled
reference response evaluated at that scaled filter's realized ratio. This
conditional lock removes direct bandwidth-induced peak drift at a fixed ratio,
but it does not force equality with the simultaneously realized baseline peak:
the baseline may have a different level-derived ratio. Consensus therefore
treats controller-induced drift as bandwidth instability, and the analysis
fails only when no positive sub-Nyquist center reaches the conditional target
within that strict continuous-frequency tolerance. This nonlinear behavior is
a GCFB-specific extension of the paper's
fixed-bandwidth-window consensus. At 1 kHz and 50 dB, the endpoint scales produce
composite-filter ERBs of approximately `0.81` and `1.19` times the baseline,
bracketing the roughly 10% to 18% between-listener variation reported for normal
hearing ([Moore et al., 1990](https://doi.org/10.1121/1.399960);
[Shen and Richards, 2013](https://doi.org/10.1121/1.4812856)). They test
stability around the configured listener rather than simulating hearing loss;
listener-specific widening is already represented by the GCFB hearing-loss and
compression-health parameters
([Irino, 2023](https://doi.org/10.1109/ACCESS.2023.3298673)). Each reassigned
energy map is normalized by its own maximum. The agreement map counts the
fraction above the relative support floor, and salience is the
required-agreement order statistic (the default requires every scale). The
returned ordinary `GcfbOutput` and the analysis at the unique `1.0` baseline
share the exact unscaled run.

These extensions are model-specific analogues of the paper's Gaussian-STFT
experiments. Its STFT-zero topology, unlimited localization precision, and
reconstruction behavior do not automatically hold for the nonlinear GCFB.
The complex output preserves phase for analysis and coherence measurement, but
it is not an invertible GCFB representation and has no synthesis guarantee.
The batch complex and consensus paths use the offline/acausal imaginary branch.
The streaming paths use causal quadrature atoms and rolling normalization. In
batch sample mode, frequency is the finite-DFT derivative of the realized
coefficient history; in streaming sample mode, it is the wrapped backward phase
increment. Both include the realized HP-AF variation without differentiating
through the nonlinear level estimator.

Render a self-contained, three-panel comparison of causal source energy, its
rolling time-frequency reassignment, and streaming bandwidth-consensus
salience with:

```bash
cargo run --example v234_reassignment_spectrogram
```

The example writes `target/v234_reassignment_spectrogram.png` by default. Pass
an alternative PNG path after `--` to choose the destination:

```bash
cargo run --example v234_reassignment_spectrogram -- /tmp/comparison.png
```

The first two panels share a -60 to 0 dB color scale referenced to their joint
maximum. The target-grid panel can contain slightly less energy because causal
coordinates outside the finite plotting grid are discarded. The third panel
shows rolling salience for the default `0.8`, `1.0`, and `1.2` bandwidth
scales, suppresses bins outside the consensus mask, and uses a separate -60 to
0 dB scale referenced to normalized salience `1.0`. The example also reports
the rolling window, active scales, agreement requirement, accepted
consensus-bin count, and retained energy.

Run the deterministic tones, clicks, chirp, and seeded-noise example to inspect
immediate causal phase coordinates and delayed rolling-consensus frames (it
only prints measurements and writes no files) with:

```bash
cargo run --example v234_phase_consensus
```

Build and test with:

```bash
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --doc
```

### Rust/Python parity properties

In addition to fixed Python-generated regression fixtures, the integration
suite generates and shrinks live differential cases with the Python v2.34
implementation as the oracle.
It covers auditory scales, level calibration, windows, convolution, cepstra,
framing, gammachirp impulse/frequency responses, asymmetric and compressive
filters, frequency conversion, field-to-cochlea transfer functions, hearing
level utilities, and the end-to-end v2.34 filterbank.

Activate a Python environment containing NumPy and SciPy, then run:

```bash
GAMMACHIRPY_REQUIRE_PYTHON=1 cargo test --test python_properties
```

If that environment's interpreter is not named `python3`, select it explicitly:

```bash
GAMMACHIRPY_PYTHON=/path/to/python \
GAMMACHIRPY_REQUIRE_PYTHON=1 \
cargo test --test python_properties
```

Without the strict switch, the live suite skips when its Python dependencies
are unavailable so that the Rust crate can still be tested on its own. CI
should always set `GAMMACHIRPY_REQUIRE_PYTHON=1` so a missing reference runtime
is reported as a failure. `PROPTEST_CASES` can raise the generated case count
for longer stress runs.

The Rust crate does not expose GCFB v2.11. The original Python modules,
notebooks, reference MATLAB outputs, figures, and instructions remain in
`gcfb_v211/` and `gcfb_v234/`; the sections below document that bundled Python
reference project, including its historical v2.11 implementation.

<div style="text-align: center">
    <img src="./figs/gammachirpy_pulse.jpg" width="425px">
</div>

## Updates
- May, 2024
    - **Add: the new version ([v234](https://github.com/kyama0321/gammachirpy/blob/main/gcfb_v234)) :tada: , a frame-based processing with hearing loss characteristics ([Irino, 2023](https://doi.org/10.1109/ACCESS.2023.3298673))** 
    - Update: some functions for the previous version ([v211](https://github.com/kyama0321/gammachirpy/blob/main/gcfb_v211)) 

## Links

- GitHub: [https://github.com/kyama0321/gammachirpy](https://github.com/kyama0321/gammachirpy)
- Documents: T.B.D.

## What is the Dynamic Compressive Gammachirp Filterbank?

- The dynamic compressive gammachirp filterbank (dcGC-FB) is a time-domain and non-linear cochlear processing model ([Irino and Patterson, 2006](https://ieeexplore.ieee.org/document/1709909)).

<div style="text-align: center">
    <img src="./figs/frequency_response.jpg" width="425px">
</div>

- The compressive gammachirp auditory filter (cGC) consists of a passive gammachirp filter (pGC) and a high-pass asymmetric function (HP-AF).
  
- The HP-AF shifts in frequency with stimulus level as dictated by data on the compression of basilar membrane motion.

<div style="text-align: center">
    <img src="./figs/cgc_pgc_hpaf.jpg" width="425px">
</div>

- The dcGC-FB contains a fast-acting level control circuit for the cGC filter, and it can explain:
  - level-dependent and asymmetric auditory filter shape
  - masking patterns and excitation patterns
  - fast compression (cochlear amplifier)
  - two-tone supression.

<div style="text-align: center">
    <img src="./figs/gc_gt_freq.jpg" width="425px">
    <img src="./figs/filter_level_dependency.jpg" width="425px">
    <img src="./figs/IO_function.jpg" width="425px">
</div>

- The Gammachirp filter explains a notched-noise masking data well for normal hearing and hearing impaired listeners ([Patterson+, 2003](https://doi.org/10.1121/1.1600720); [Matsui+, 2016](https://asa.scitation.org/doi/10.1121/1.4970396)).

- **The new version (gcfb_v234) includes hearing loss characteristics ([Irino, 2023](https://doi.org/10.1109/ACCESS.2023.3298673))**
    - **audiogram with a compression factor $\alpha$**    
    - **input/output function with a compression factor $\alpha$ and a audiogram**
    - **filter outputs based frame-based processing**

<div style="text-align: center">
    <img src="./figs/audiograms.jpg" width="425px">
    <img src="./figs/audiogram_hl3_compression_05.jpg" width="425px">
    <img src="./figs/IO_function_NH_HL3.jpg" width="425px">
    <img src="./figs/gammachirpy_speech_NH_HL3_40dbspl.jpg" width="425px">
</div>

- The MATLAB packages of the original Gammachirp filterbank are [HERE](https://github.com/AMLAB-Wakayama/gammachirp-filterbank).

## About the GammachirPy Project

- The project name, "GammachirPy (がんまちゃーぴー)" is "Gammachirp + Python".

- This project aims to translate the original MATLAB codes to Python and share them as an open-source software ([Apache-2.0 license](https://github.com/kyama0321/gammachirpy/blob/main/LICENSE.md)).
  
- I have made some demo scripts of the Jupyter Notebook for educational uses. You can also open and execute them of Google Colaboratory. See **gcfb_v211/demo_*.ipynb** and **gcfb_v234/demo_*.ipynb** files.

## Repository Structure

- The directory structure is almost the same as the original MATLAB page.
  - **[gcfb_v211](https://github.com/kyama0321/gammachirpy/tree/main/gcfb_v211)**: sample-by-sample processing version
  - **[gcfb_v234](https://github.com/kyama0321/gammachirpy/tree/main/gcfb_v234)**: a new frame-based processing version for Wadai Hearing Impaired Simulator (WHIS)

- In each version, the directory mainly contains:
  - **gcfb_v\*.py**: dynamic compressive gammachirp (dcGC) filter
  - **gammachirp.py**: passive gammachirp (pGC) filter
  - **utils.py**: useful functions for auditory signal processing
  - **test_gcfb_v\*_{pulse/speech}.py**: test and demo scripts for practical uses as a plain Python file.
  - **demo_gcfb_v\*_{pulse/speech}.ipynb**: demo scripts for practical uses on the Jupyter Notebook. The scripts are based on test_gcfb_v*_{pulse/speech}.py.
  - **demo_gammachirp.ipynb**: demo scripts for educational uses of the dcGC-FB on the Jupyter Notebook
  - **/sample**: a speech file
  - **/original**: outputs of original Gammachirp filterbank (*.mat) 

## Requirements

- Python >= 3.11.1
- NumPy >= 1.26.4
- SciPy >= 1.10.1
- Matplotlib >= 3.7.1
- Jupyter >= 1.0.0

Please see more information in [requirements.txt](https://github.com/kyama0321/gammachirpy/blob/main/requirements.txt).

## Installation

1. fork/clone the gammachirpy repository
    ```bash
    git clone https://github.com/kyama0321/gammachirpy
    cd gammachirpy
    ```

2. If you use "venv":
    ```bash
    python3.11 -m venv venv
    . venv/bin/activate
    pip install --upgrade pip
    # pip install setuptools # if you fail to install packages
    pip install -r requirements.txt
    ```

## Getting Started

Please see the README file in each directory.

  - **[gcfb_v211](https://github.com/kyama0321/gammachirpy/tree/main/gcfb_v211)**: sample-by-sample processing version
  - **[gcfb_v234](https://github.com/kyama0321/gammachirpy/tree/main/gcfb_v234)**: a new frame-based processing version for Wadai Hearing Impaired Simulator (WHIS)

## Reproducibility

- In the **[gcfb_v234/demo_gammachirp.ipynb](https://github.com/kyama0321/gammachirpy/blob/main/gcfb_v234/demo_gammachirp.ipynb)** and **[gcfb_v211/demo_gammachirp.ipynb](https://github.com/kyama0321/gammachirpy/blob/main/gcfb_v211/demo_gammachirp.ipynb)**, essential characteristics of the gammachirp filterbank are explained and checked with the GammachirPy package.

    - v211: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyama0321/gammachirpy/blob/main/gcfb_v211/demo_gammachirp.ipynb)
    - v234: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyama0321/gammachirpy/blob/main/gcfb_v234/demo_gammachirp.ipynb)


- In the **[gcfb_v234/demo_gcfb_v234_pulse.ipynb](https://github.com/kyama0321/gammachirpy/blob/main/gcfb_v234/demo_gcfb_v234_pulse.ipynb)** and **[gcfb_v211/demo_gcfb_v211_pulse.ipynb](https://github.com/kyama0321/gammachirpy/blob/main/gcfb_v211/demo_gcfb_v211_pulse.ipynb)**, a simple pulse train are used as an input signal with some sound pressure levels (SPLs) to compare outputs of the GammachirPy and the original Gammachirp.

<div style="text-align: center">
    <img src="./figs/gammachirpy_gammachirp.jpg" width="425px">
</div>

- In the latest release, the root-mean-squared error (RMSE) between output signals (cgc_out) of the GammachirPy and the original Gammachirp in each level is:

    - v211 (sample-by-sample processing)

       [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyama0321/gammachirpy/blob/main/gcfb_v211/demo_gcfb_v211_pulse.ipynb)

        | gcfb | SPL (dB) | RMSE |
        | --- | --- | --- |
        | v211 | 40 | 4.11e-14 |
        | v211 | 60 | 2.25e-13 |
        | v211 | 80 | 1.73e-12 |

    - v234 (frame-based processing with hearing loss characteristics)
      - NH: normal hearing
      - HL3: hearing loss type #3 (ISO-7029; 70 year old, male)

      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyama0321/gammachirpy/blob/main/gcfb_v234/demo_gcfb_v234_pulse.ipynb)

        | gcfb | NH/HL | SPL (dB) | RMSE |
        | --- | ---  | --- | --- |
        | v234 | NH | 40 | 1.10e-16|
        | v234 | NH | 60 | 7.42e-16|
        | v234 | NH | 80 | 4.08e-15|
        | v234 | HL3 | 40 | 4.14e-17|
        | v234 | HL3 | 60 | 3.06e-16|
        | v234 | HL3 | 80 | 2.04e-15|

- There are still small errors between the GammachirPy and the original Gammachirp. I would like to improve them with code refactorings in the future:-)

## :warning: Compatibility Note :warning:
- Please be aware that the original [firpm](https://jp.mathworks.com/help/signal/ref/firpm.html) function in MATLAB is incompatible with the [scipy.signal.remez](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.remez.html) function in GammachirPy code and the [firmpm](https://octave.sourceforge.io/signal/function/firpm.html) function in Octave due to the specifications of input arguments. The function is used for outer and middle ear corrections (`utils.out_mid_crct_filt`). You can use the frequency correction with options such as "ELC," "FreeField," and "EarDrum," but the phase characteristics are slightly different from the original Gammachirp outputs. 

## Acknowledgements

The packages is inspired from [AMLAB-Wakayama/gammachirp-filterbank](https://github.com/AMLAB-Wakayama/gammachirp-filterbank) by [Prof. Toshio Irino](https://web.wakayama-u.ac.jp/~irino/index-e.html), Auditory Media Laboratory, Wakayama University, Japan.

## References

- [R. D. Patterson, M. Unoki, and T. Irino "Extending the domain of center frequencies for the compressive gammachirp auditory filter," J. Acoust. Soc. Am., 114 (3), pp.1529-1542, 2003.](https://doi.org/10.1121/1.1600720)
- [T. Irino and R. D. Patterson, "A dynamic compressive gammachirp auditory filterbank" IEEE Trans. Audio, Speech, and Language Process., 14(6), pp.2222-2232, 2006.](https://doi.org/10.1109/TASL.2006.874669)
- [T. Irino, "An introduction to auditory filter," J. Acoust. Soc. Jpn., 66(10), pp.505-512, 2010. (in Japanese)](https://doi.org/10.20697/jasj.66.10_506)
- [T. Matsui, T. Irino, H. Inabe, Y. Nishimura and R. D. Patterson, "Estimation of auditory compression and filter shape of elderly listeners using notched noise masking," J. Acoust. Soc. Am., 140, p.3274, 2016.](https://asa.scitation.org/doi/10.1121/1.4970396)
- [T. Irino and R. D. Patterson, "The gammachirp auditory filter and its application to speech perception," Acoust. Sci. & Tech., 41(1), pp.99-107, 2020.](https://doi.org/10.1250/ast.41.99)
- [T. Irino, “Hearing Impairment Simulator Based on Auditory Excitation Pattern Playback: WHIS,” IEEE Access, 11, pp. 78419–78430, 2023.](https://doi.org/10.1109/ACCESS.2023.3298673)
