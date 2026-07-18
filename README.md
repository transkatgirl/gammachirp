# GammachirPy / `gammachirpy` Rust crate

Dynamic compressive gammachirp filterbanks, now available as a Rust crate as
well as the original Python implementation.

## Rust crate

The Rust port keeps the original versioned structure:

- `gcfb_v211::{gammachirp, utils, gcfb_v211}` implements sample-by-sample
  processing.
- `gcfb_v234::{gammachirp, utils, gcfb_v234}` implements frame processing,
  audiograms, cochlear hearing loss, synthesis, and envelope-modulation tools.

Filterbank matrices use the same channel-major orientation as Python: rows are
channels and columns are samples or frames. Parameters are typed structs, and
invalid inputs return `gammachirpy::Result`.

```rust
use gammachirpy::gcfb_v234::{GcParam, gcfb_v234};

let input = [1.0, 0.0, 0.0, 0.0];
let parameters = GcParam {
    num_ch: 32,
    out_mid_crct: "No".into(),
    ..GcParam::default()
};

let output = gcfb_v234(&input, parameters)?;
assert_eq!(output.dcgc_out.nrows(), 32);
# Ok::<(), gammachirpy::Error>(())
```

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
use gammachirpy::gcfb_v234::{GcParam, gcfb_v234_with_reassignment};

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
# Ok::<(), gammachirpy::Error>(())
```

The imaginary analysis branch is offline and acausal; the ordinary real GCFB
branch remains causal and unchanged. In frame mode, dynamic compression is a
positive real gain, so it affects transported energy but not coordinates. In
sample mode, all three operator applications replay the recorded HP-AF
coefficient history. Its exactness is conditional on that history: the
analysis does not differentiate through the nonlinear level estimator that
created the coefficients. In all modes, exactness refers to the implemented
zero-padded finite discrete operator up to floating-point error. The selected
DFT length is exposed as `ReassignmentResult::analysis_fft_len`.

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

before bilinear deposition. `PhaseReassignmentResult::complex_map` retains
absolute phase for analysis, while `phase_coherence_map` is the magnitude of
the complex sum divided by the sum of contribution magnitudes. Empty bins have
zero coherence. The matched `unreassigned_energy_map` contains only the same
floor- and boundary-accepted contributions on the reassigned map's grid. Its
time coordinate is the source sample for sample-based analyses and the
originating frame for frame-based analyses.

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
`lvl_est.b2`; chirp, compression, level-control, hearing-loss, and frequency-
grid parameters remain fixed. At 1 kHz and 50 dB, the endpoint scales produce
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
All complex and consensus paths use the offline/acausal imaginary branch. In
sample mode they replay the realized coefficient history, so their meaning is
conditional on that history and does not include differentiation through the
nonlinear estimator.

Render a self-contained, three-panel comparison of the matched energy map,
its time-frequency reassignment, and the default bandwidth-consensus salience
with:

```bash
cargo run --example v234_reassignment_spectrogram
```

The example writes `target/v234_reassignment_spectrogram.png` by default. Pass
an alternative PNG path after `--` to choose the destination:

```bash
cargo run --example v234_reassignment_spectrogram -- /tmp/comparison.png
```

The first two panels use the same retained analytic energy and the same -60 to
0 dB color scale, so their visual difference reflects reassignment rather than
per-panel normalization. The third panel shows salience for the default `0.8`,
`1.0`, and `1.2` bandwidth scales, suppresses bins outside the consensus mask,
and uses a separate -60 to 0 dB scale referenced to normalized salience `1.0`.
The example also reports the active scales, agreement requirement, and accepted
consensus-bin count and fraction.

Run the deterministic tones, clicks, chirp, and seeded-noise example (it only
prints measurements and writes no files) with:

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
suite generates and shrinks live differential cases with Python as the oracle.
It covers auditory scales, level calibration, windows, convolution, cepstra,
framing, gammachirp impulse/frequency responses, asymmetric and compressive
filters, frequency conversion, field-to-cochlea transfer functions, hearing
level utilities, and both end-to-end filterbanks.

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

The original Python modules, notebooks, reference MATLAB outputs, figures, and
instructions remain in `gcfb_v211/` and `gcfb_v234/`.

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
