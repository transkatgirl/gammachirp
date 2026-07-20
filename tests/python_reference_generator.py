#!/usr/bin/env python3
"""Regenerate golden values used by the Rust/Python parity tests.

Run this from the repository root with a Python environment containing NumPy,
SciPy, and Matplotlib.  The checked-in fixture lets normal ``cargo test`` runs
remain independent of Python and those comparatively heavy dependencies.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "fixtures" / "python_reference.json"


def encoded(value):
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "values": array.astype(float).reshape(-1).tolist(),
    }


def load_version(name):
    directory = ROOT / name
    sys.path.insert(0, str(directory))
    utils = importlib.import_module("utils")
    gammachirp = importlib.import_module("gammachirp")
    filterbank = importlib.import_module(name)
    return directory, utils, gammachirp, filterbank


def unload_version(directory):
    sys.path.remove(str(directory))
    for name in ("utils", "gammachirp", "gcfb_v211", "gcfb_v234"):
        sys.modules.pop(name, None)


class Param:
    pass


def make_v211_param(control):
    param = Param()
    param.fs = 8000
    param.num_ch = 4
    param.f_range = np.array([200.0, 1500.0])
    param.out_mid_crct = "No"
    param.ctrl = control
    return param


def make_v234_param(control, processing="frame-base", hearing_loss="NH"):
    param = Param()
    param.fs = 8000
    param.num_ch = 4
    param.f_range = np.array([200.0, 1500.0])
    param.out_mid_crct = "No"
    param.ctrl = control
    param.dyn_hpaf_str_prc = processing
    param.hloss_type = hearing_loss
    if hearing_loss == "HL3":
        param.hloss_compression_health = 0.5
    return param


def selected_rows(values, indices):
    values = np.asarray(values)
    return encoded(values[:, indices])


def real_scalar(value):
    return float(np.real(np.asarray(value).reshape(-1)[0]))


def v211_references():
    directory, utils, gc, gcfb = load_version("gcfb_v211")
    result = {}

    signal = np.array([-3.0, -1.0, 2.0, 4.0])
    result["utility_scalars"] = {
        "rms": float(utils.rms(signal)),
        "nextpow2": [utils.nextpow2(n) for n in (1, 2, 3, 15, 16, 17, 1000)],
        "freq2mel": encoded(utils.freq2mel(np.array([50.0, 100.0, 1000.0, 6000.0]))),
        "mel2freq": encoded(utils.mel2freq(np.array([75.0, 500.0, 1000.0, 2500.0]))),
    }
    erb_rate, erb_width = utils.freq2erb(np.array([0.0, 50.0, 100.0, 1000.0, 6000.0]))
    frequencies, inverse_width = utils.erb2freq(erb_rate)
    result["erb"] = {
        "rate": encoded(erb_rate),
        "width": encoded(erb_width),
        "inverse_frequency": encoded(frequencies),
        "inverse_width": encoded(inverse_width),
    }
    equal_freq, equal_erb = utils.equal_freq_scale("ERB", 6, np.array([100.0, 6000.0]))
    result["equal_erb"] = {"frequency": encoded(equal_freq), "erb": encoded(equal_erb)}

    equalized, level = utils.eqlz2meddis_hc_level(signal, 63.0)
    result["level_equalization"] = {"signal": encoded(equalized), "level": encoded(level)}
    result["windows"] = {}
    for kind in ("HAM", "HAN", "BLA", "LINE"):
        window, name = utils.taper_window(12, kind, 4)
        result["windows"][kind] = {"name": name, "values": encoded(window)}
    result["fftfilt"] = encoded(
        utils.fftfilt(np.array([0.25, -0.5, 1.5, 0.75]), np.array([1.0, -2.0, 0.5, 3.0, -1.0]))
    )
    # The source implementation's homomorphic window only supports odd input
    # lengths; keep the parity fixture inside that documented behavior.
    for name, source in (
        ("odd", np.array([1.0, 0.5, -0.25, 0.125, 0.75, -0.4, 0.2])),
    ):
        cepstrum, minimum = utils.rceps(source)
        result.setdefault("rceps", {})[name] = {
            "cepstrum": encoded(cepstrum),
            "minimum_phase": encoded(minimum),
        }
    frames, centers = gcfb.set_frame4time_sequence(np.arange(1.0, 10.0), 6, 3)
    result["frames"] = {"values": encoded(frames), "centers": encoded(centers)}
    result["outer_middle_ear_tables"] = {}
    for kind in ("ELC", "MAF"):
        power, frequency, db = utils.out_mid_crct(kind, 0, 48000, 0)
        result["outer_middle_ear_tables"][kind] = {
            "power": encoded(power),
            "frequency": encoded(frequency),
            "db": encoded(db),
        }

    result["gammachirp_impulse"] = {}
    # The Python impulse routine only broadcasts reliably for one frequency at
    # a time when the chirp coefficient is nonzero.
    for frequency in (500.0, 1500.0):
        impulse = gc.gammachirp(
            frequency, 8000, 4, 1.019, -2.0, 0.3, "cos", "no"
        )
        result["gammachirp_impulse"][str(int(frequency))] = {
            "gc": encoded(impulse[0]),
            "length": encoded(impulse[1]),
            "peak_frequency": encoded(impulse[2][0]),
            "instantaneous_frequency": encoded(impulse[3]),
        }
    peak_normalized = gc.gammachirp(1000.0, 8000, 4, 1.019, -2.0, 0.0, "cos", "peak")
    envelope = gc.gammachirp(1000.0, 8000, 4, 1.019, -2.0, 0.0, "env", "no")
    sine = gc.gammachirp(1000.0, 8000, 4, 1.019, -2.0, 0.0, "sin", "no")
    result["gammachirp_carriers"] = {
        "peak_normalized": encoded(peak_normalized[0]),
        "envelope": encoded(envelope[0]),
        "sine": encoded(sine[0]),
    }

    response = gc.gammachirp_frsp(
        np.array([250.0, 1000.0, 3000.0]), 8000, 4, 1.019, -2.0, 0.25, 256
    )
    bins = [0, 1, 5, 17, 64, 127, 255]
    result["gammachirp_response"] = {
        "bins": bins,
        "frequency": encoded(response[1][bins, 0]),
        "amplitude": selected_rows(response[0], bins),
        "peak_frequency": encoded(response[2]),
        "group_delay": selected_rows(response[3], bins),
        "phase": selected_rows(response[4], bins),
    }

    coefficients = gcfb.make_asym_cmp_filters_v2(
        8000,
        np.array([[500.0], [1500.0]]),
        np.array([[2.17], [1.8]]),
        np.array([[2.2], [1.5]]),
    )
    result["asymmetric_coefficients"] = {
        "ap": encoded(coefficients.ap),
        "bz": encoded(coefficients.bz),
    }
    # Discrete component responses used by the exact peak-lock model.  These
    # intentionally sample the implemented cosine FIR and digital biquads,
    # rather than either analytic gammachirp response surrogate.
    peak_fft_len = 65536
    peak_bins = np.array([0, 1, 257, 4096, 8192, 16000, 32768])
    pgc_spectrum = np.fft.rfft(np.asarray(peak_normalized[0]).reshape(-1), peak_fft_len)
    z = np.exp(-2j * np.pi * peak_bins / peak_fft_len)
    acf_power = np.empty((2, len(peak_bins)))
    for channel in range(2):
        response_values = np.ones(len(peak_bins), dtype=complex)
        for section in range(4):
            numerator = sum(coefficients.bz[channel, tap, section] * z**tap for tap in range(3))
            denominator = sum(coefficients.ap[channel, tap, section] * z**tap for tap in range(3))
            response_values *= numerator / denominator
        acf_power[channel, :] = np.abs(response_values) ** 2
    result["discrete_component_response"] = {
        "fft_len": peak_fft_len,
        "bins": peak_bins.tolist(),
        "pgc_power": encoded(np.abs(pgc_spectrum[peak_bins]) ** 2),
        "acf_power": encoded(acf_power),
    }
    acf_response = gcfb.asym_cmp_frsp_v2(
        np.array([500.0, 1500.0]), 8000, np.array([2.17, 1.8]), np.array([2.2, 1.5]), 256, 4
    )
    result["asymmetric_response"] = {
        "bins": bins,
        "acf": selected_rows(acf_response[0], bins),
        "frequency": encoded(acf_response[1][bins]),
        "function": selected_rows(acf_response[2], bins),
    }
    compressed = gcfb.cmprs_gc_frsp(
        np.array([500.0, 1500.0]), 8000, 4, np.array([[1.81], [1.7]]),
        np.array([[-2.96], [-2.5]]), np.array([[0.9], [1.05]]),
        np.array([[2.17], [1.9]]), np.array([[2.2], [1.8]]), 256
    )
    result["compressed_response"] = {
        "bins": bins,
        "pgc": selected_rows(compressed.pgc_frsp, bins),
        "cgc": selected_rows(compressed.cgc_frsp, bins),
        "normalized": selected_rows(compressed.cgc_nrm_frsp, bins),
        "fp1": encoded(compressed.fp1),
        "fr2": encoded(compressed.fr2),
        "fp2": encoded(compressed.fp2),
        "peak_value": encoded(compressed.val_fp2),
        "normalization": encoded(compressed.norm_fct_fp2),
    }

    conversions = []
    for fr1 in (250.0, 1000.0, 2500.0):
        fp2, fr2 = gcfb.fr1_to_fp2(4, 1.81, -2.96, 2.17, 2.2, 0.95, np.array([fr1]))
        inverse_fr1, inverse_fp1 = gcfb.fp2_to_fr1(
            4, 1.81, -2.96, 2.17, 2.2, 0.95, real_scalar(fp2)
        )
        conversions.append([
            fr1, real_scalar(fp2), real_scalar(fr2),
            real_scalar(inverse_fr1), real_scalar(inverse_fp1),
        ])
    result["frequency_conversions"] = encoded(conversions)

    _, status = gcfb.acfilterbank(coefficients, [])
    acf_outputs = []
    for sample in ([1.0, -0.5], [0.0, 0.25], [-0.75, 0.0], [0.5, 1.0], [0.0, 0.0], [0.25, -0.25]):
        output, status = gcfb.acfilterbank(coefficients, status, np.asarray(sample), 0)
        acf_outputs.append(output[:, 0])
    result["asymmetric_filter_sequence"] = encoded(acf_outputs)

    set_param, set_response = gcfb.set_param(make_v211_param("dynamic"))
    result["set_param"] = {
        "fr1": encoded(set_response.fr1),
        "erb_space": float(set_response.erb_space1),
        "ef": encoded(set_response.ef),
        "b1": encoded(set_response.b1_val),
        "c1": encoded(set_response.c1_val),
        "fp1": encoded(set_response.fp1),
        "b2": encoded(set_response.b2_val),
        "c2": encoded(set_response.c2_val),
        "exp_decay": float(set_param.lvl_est.exp_decay_val),
        "channel_shift": float(set_param.lvl_est.n_ch_shift),
        "level_channels": encoded(set_param.lvl_est.n_ch_lvl_est - 1),
        "linear_minimum": float(set_param.lvl_est.lvl_lin_min_lim),
        "linear_reference": float(set_param.lvl_est.lvl_lin_ref),
    }

    input_signal = np.array([1.0, -0.25, 0.5, 0.0, -0.1, 0.2] + [0.0] * 26)
    result["filterbank"] = {}
    for control in ("static", "dynamic"):
        cgc, pgc, param, response = gcfb.gcfb_v211(input_signal, make_v211_param(control))
        result["filterbank"][control] = {
            "cgc": encoded(cgc),
            "pgc": encoded(pgc),
            "fr2": encoded(getattr(response, "fr2", [])),
            "frat": encoded(getattr(response, "frat_val", [])),
            "level_db": encoded(getattr(response, "lvl_db", [])),
            "gain": encoded(getattr(response, "gain_factor", [])),
        }

    unload_version(directory)
    return result


def v234_references():
    directory, utils, _gc, gcfb = load_version("gcfb_v234")
    result = {}

    precise_input = np.array([0.25, -0.5, 0.75, -1.0])
    precise_signal, precise_level = utils.eqlz2meddis_hc_level(precise_input, [], 94.0)
    result["precise_level_equalization"] = {
        "signal": encoded(precise_signal),
        "level": encoded(precise_level),
    }
    result["interpolation"] = {
        "inside": encoded(utils.interp1([0.0, 1.0, 4.0], [0.0, 2.0, -1.0], [0.25, 2.5])),
        "extrapolated": encoded(utils.interp1([0.0, 1.0, 4.0], [0.0, 2.0, -1.0], [-2.0, 6.0], extrapolate=True)),
    }
    result["hearing_level"] = {
        "spl": [float(np.asarray(utils.hl2spl(f, h)).reshape(-1)[0]) for f, h in ((125, 10), (1000, 20), (4000, 35), (8000, 5))],
        "cochlea": [
            float(np.asarray(utils.hl2pin_cochlea(np.asarray(f), h)).reshape(-1)[0])
            for f, h in ((125, 10), (1000, 20), (4000, 35), (8000, 5))
        ],
    }
    middle_frequency, middle_db = utils.trans_func_middle_ear_moore16()
    result["middle_ear_table"] = {"frequency": encoded(middle_frequency), "db": encoded(middle_db)}
    result["field_tables"] = {}
    for kind in ("FreeField", "DiffuseField"):
        frequency, db = utils.trans_func_free_field2eardrum_moore16(kind)
        result["field_tables"][kind] = {"frequency": encoded(frequency), "db": encoded(db)}
    itu_frequency, itu_db = utils.trans_func_free_field2eardrum_itu("ITU")
    result["field_tables"]["ITU"] = {"frequency": encoded(itu_frequency), "db": encoded(itu_db)}

    transfer_param = utils.param_trans_func(
        fs=16000, n_frq_rsl=64, freq_calib=1000,
        type_field2eardrum="FreeField", type_midear2cochlea="MiddleEar"
    )
    transfer, _ = utils.trans_func_field2cochlea(transfer_param)
    result["field_to_cochlea"] = {
        "frequency": encoded(transfer.freq),
        "field_db": encoded(transfer.field2eardrum_db),
        "middle_db": encoded(transfer.midear2cochlea_db),
        "total_db": encoded(transfer.field2cochlea_db),
        "frequency_calibration": float(transfer.freq_calib),
        "field_at_calibration": float(transfer.field2eardrum_db_at_freq_calib),
        "field_compensation": float(transfer.field2eardrum_db_cmpnst_db),
        "middle_at_calibration": float(transfer.midear2cochlea_db_at_freq_calib),
        "total_at_calibration": float(transfer.field2cochlea_db_at_freq_calib),
    }
    floor_source = np.array([[-0.1, 0.0, 0.02], [0.1, 0.5, 1.0]])
    result["absolute_threshold_scaling"] = {
        "none": encoded(utils.eqlz_gcfb2rms1_at_0db(floor_source)),
        "zero": encoded(utils.eqlz_gcfb2rms1_at_0db(floor_source, "ZeroFloor")),
    }

    result["set_param"] = {}
    for hearing in ("NH", "HL3"):
        param, response = gcfb.set_param(make_v234_param("dynamic", hearing_loss=hearing))
        result["set_param"][hearing] = {
            "fr1": encoded(response.fr1),
            "fp1": encoded(response.fp1),
            "frat0": encoded(response.frat0_val),
            "frat1": encoded(response.frat1_val),
            "pc": encoded(response.pc_hpaf),
            "frat0_pc": encoded(response.frat0_pc),
            "window": encoded(param.dyn_hpaf.val_win),
            "hearing_level": encoded(param.hloss.hearing_level_db),
            "active_loss": encoded(param.hloss.pin_loss_db_act),
            "initial_active_loss": encoded(param.hloss.pin_loss_db_act_init),
            "passive_loss": encoded(param.hloss.pin_loss_db_pas),
            "health": encoded(param.hloss.compression_health),
            "gain_compensation": encoded(param.hloss.af_gain_cmpnst_db),
            "fb_hearing_level": encoded(param.hloss.fb_hearing_level_db),
            "fb_active_loss": encoded(param.hloss.fb_pin_loss_db_act),
            "fb_passive_loss": encoded(param.hloss.fb_pin_loss_db_pas),
            "fb_health": encoded(param.hloss.fb_compression_health),
            "fb_gain_compensation": encoded(param.hloss.fb_af_gain_cmpnst_db),
        }

    param, response = gcfb.set_param(make_v234_param("dynamic"))
    pins = np.array([-20.0, 0.0, 30.0, 60.0, 90.0, 120.0])
    asym, io, _ = gcfb.gcfb_v23_asym_func_in_out(param, response, 1000.0, 0.65, pins)
    result["asymmetric_io"] = {
        "pins": encoded(pins),
        "asymmetry_db": encoded(asym),
        "output_db": encoded(io),
        "inverse": encoded([
            gcfb.gcfb_v23_asym_func_in_out_inv_io_func(param, response, 1000.0, 0.65, value)
            for value in io
        ]),
    }

    modulation_param = Param()
    modulation_param.fs = 2000.0
    modulation_param.fc_mod_list = np.array([1.0, 4.0, 16.0, 64.0])
    modulation_input = np.array([1.0, 0.5, -0.25, 0.75, -1.0, 0.0, 0.2, -0.1, 0.3, 0.0])
    result["modulation_filterbank"] = encoded(gcfb.gcfb_v23_env_mod_fb(modulation_input, modulation_param))

    envelope_frames = np.empty((4, 16))
    for channel in range(4):
        for frame in range(16):
            baseline = 1.25 + 0.35 * channel
            modulation = (0.15 + 0.04 * channel) * np.sin(
                2 * np.pi * (channel + 1) * frame / 16
            )
            envelope_frames[channel, frame] = (
                baseline + modulation + (0.2 if frame == channel + 4 else 0.0)
            )
    envelope_param, _ = gcfb.set_param(make_v234_param("dynamic"))
    # The reference set_param keeps channel frequencies as a column vector;
    # flatten it for the scalar-per-channel envelope routine.
    envelope_param.fr1 = np.asarray(envelope_param.fr1).reshape(-1)
    envelope_modulation_param = Param()
    envelope_modulation_param.reduce_db = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0])
    envelope_modulation_param.f_cutoff = np.array([32.0, 48.0, 64.0, 96.0, 128.0, 160.0, 192.0])
    envelope_output, envelope_modulation_param = gcfb.gcfb_v23_env_mod_loss(
        envelope_frames, envelope_param, envelope_modulation_param
    )
    result["envelope_modulation_loss"] = {
        "output": encoded(envelope_output),
        "fs": float(envelope_modulation_param.fs),
        "fb_fr1": encoded(envelope_modulation_param.fb_fr1),
        "fb_reduce_db": encoded(envelope_modulation_param.fb_reduce_db),
        "fb_f_cutoff": encoded(envelope_modulation_param.fb_f_cutoff),
    }

    input_signal = np.array([1.0, -0.25, 0.5, 0.0, -0.1, 0.2] + [0.0] * 26)
    result["filterbank"] = {}
    # The source's static path refers to gc_param.frat0_pc/frat1_val even
    # though set_param places them on gc_resp. Its sample path similarly reads
    # a missing dyn_hpaf.type field. Keep end-to-end goldens on the executable
    # frame path; Rust-only unit tests cover its working static/sample paths.
    cases = (
        ("dynamic_frame", "dynamic", "frame-base", "NH"),
        ("dynamic_frame_hl3", "dynamic", "frame-base", "HL3"),
    )
    for name, control, processing, hearing in cases:
        dcgc, scgc, param, response = gcfb.gcfb_v234(
            input_signal, make_v234_param(control, processing, hearing)
        )
        result["filterbank"][name] = {
            "dcgc": encoded(dcgc),
            "scgc": encoded(scgc),
            "fr2": encoded(response.fr2),
            "frat": encoded(response.frat_val),
            "level_db": encoded(response.lvl_db),
            "level_db_frame": encoded(getattr(response, "lvl_db_frame", [])),
            "pgc_frame": encoded(getattr(response, "pgc_frame", [])),
            "scgc_frame": encoded(getattr(response, "scgc_frame", [])),
            "asymmetry_gain": encoded(getattr(response, "asym_func_gain", [])),
            "gain": encoded(getattr(
                response,
                "gain_factor",
                10 ** (-(param.hloss.fb_af_gain_cmpnst_db) / 20),
            )),
        }

    synthesis_input = np.arange(1.0, 25.0).reshape(4, 6) / 10.0
    synthesis_param = make_v234_param("dynamic")
    synthesis_param, _ = gcfb.set_param(synthesis_param)
    synthesis_param.out_mid_crct = "NO"
    result["synthesis_no_correction"] = encoded(gcfb.gcfb_v23_synth_snd(synthesis_input, synthesis_param))

    unload_version(directory)
    return result


def main():
    references = {
        "metadata": {
            "implementation": "checked-in GammachirPy Python modules",
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "v211": v211_references(),
        "v234": v234_references(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(references, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
