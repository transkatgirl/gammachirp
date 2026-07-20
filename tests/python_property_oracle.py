#!/usr/bin/env python3
"""Streaming oracle for the Rust/Python differential property tests.

The protocol is one JSON object per line on stdin and stdout.  Keeping this
process alive makes hundreds of generated comparisons much cheaper than
starting Python (and importing SciPy) for every property-test case.
"""

from __future__ import annotations

import importlib
import contextlib
import builtins
import io
import json
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]


class Param:
    pass


def encoded(value):
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError("the Rust API has no complex-valued output counterpart")
    return {
        "shape": list(array.shape),
        "values": array.astype(float).reshape(-1).tolist(),
    }


def load_version(name):
    directory = ROOT / name
    sys.path.insert(0, str(directory))
    try:
        utils = importlib.import_module("utils")
        gammachirp = importlib.import_module("gammachirp")
        filterbank = importlib.import_module(name)
        return utils, gammachirp, filterbank
    finally:
        sys.path.remove(str(directory))
        for module in ("utils", "gammachirp", "gcfb_v234"):
            sys.modules.pop(module, None)


def make_v234_param(request):
    param = Param()
    param.fs = request["fs"]
    param.num_ch = request["channels"]
    param.f_range = np.asarray(request["f_range"], dtype=float)
    param.out_mid_crct = "No"
    param.ctrl = "dynamic"
    param.dyn_hpaf_str_prc = "frame-base"
    param.hloss_type = request["hearing_loss"]
    if param.hloss_type == "HL3":
        param.hloss_compression_health = 0.5
    return param


def digital_compressive_response(frequencies, fs, order, b1, c1, ratio, b2, c2, bins):
    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    b1 = np.asarray(b1, dtype=float).reshape(-1)
    c1 = np.asarray(c1, dtype=float).reshape(-1)
    ratio = np.asarray(ratio, dtype=float).reshape(-1)
    b2 = np.asarray(b2, dtype=float).reshape(-1)
    c2 = np.asarray(c2, dtype=float).reshape(-1)
    _, erb_width = u234.freq2erb(frequencies)
    fp1 = frequencies + c1 * np.asarray(erb_width).reshape(-1) * b1 / order
    fr2 = ratio * fp1
    acf, response_frequency, asymmetry = f234.asym_cmp_frsp_v2(
        fr2, fs, b2, c2, bins, 4
    )
    acf = np.asarray(acf)
    pgc = np.empty((len(frequencies), bins))
    transform_length = 2 * bins
    for channel, frequency in enumerate(frequencies):
        impulse = np.asarray(gc234.gammachirp(
            frequency, fs, order, b1[channel], c1[channel], 0.0, "cos", "peak"
        )[0]).reshape(-1)
        folded = np.zeros(transform_length)
        np.add.at(folded, np.arange(len(impulse)) % transform_length, impulse)
        pgc[channel, :] = np.abs(np.fft.fft(folded)[:bins])
    cgc = pgc * acf
    peak_bins = np.argmax(cgc, axis=1)
    peak_value = cgc[np.arange(len(frequencies)), peak_bins]
    normalization = 1.0 / peak_value
    normalized = cgc * normalization[:, None]
    response_frequency = np.asarray(response_frequency).reshape(-1)
    return {
        "pgc": pgc,
        "cgc": cgc,
        "normalized": normalized,
        "acf": acf,
        "asymmetry": np.asarray(asymmetry),
        "fp1": fp1,
        "fr2": fr2,
        "fp2": response_frequency[peak_bins],
        "peak_value": peak_value,
        "normalization": normalization,
        "frequency": response_frequency,
    }


def realized_frame_normalization(param, response):
    fft_len = 65536
    bins = fft_len // 2
    frequencies = np.asarray(response.fr1).reshape(-1)
    b1 = np.asarray(response.b1_val).reshape(-1)
    c1 = np.asarray(response.c1_val).reshape(-1)
    ratio = np.full(len(frequencies), param.lvl_est.frat)
    b2 = np.full(len(frequencies), param.lvl_est.b2)
    c2 = np.asarray(param.hloss.fb_compression_health).reshape(-1) * param.lvl_est.c2
    fp1 = np.asarray(response.fp1).reshape(-1)
    fr2 = ratio * fp1
    acf = np.asarray(f234.asym_cmp_frsp_v2(fr2, param.fs, b2, c2, bins, 4)[0])
    values = np.empty(len(frequencies))
    for channel, frequency in enumerate(frequencies):
        impulse = np.asarray(gc234.gammachirp(
            frequency, param.fs, param.n, b1[channel], c1[channel], 0.0, "cos", "peak"
        )[0]).reshape(-1)
        passive = np.abs(np.fft.fft(impulse, fft_len)[:bins])
        values[channel] = np.max(passive * acf[channel, :])
    return 1.0 / values


def scales(request):
    frequencies = np.asarray(request["frequencies"], dtype=float)
    mel = np.asarray(request["mel"], dtype=float)
    signal = np.asarray(request["signal"], dtype=float)
    rate, width = u234.freq2erb(frequencies)
    inverse, inverse_width = u234.erb2freq(rate)
    equal_frequency, equal_scale = u234.equal_freq_scale(
        request["scale"], request["channels"], np.asarray(request["range"], dtype=float)
    )
    return {
        "rms": float(u234.rms(signal)),
        "nextpow2": int(u234.nextpow2(request["integer"])),
        "freq2mel": encoded(u234.freq2mel(frequencies)),
        "mel2freq": encoded(u234.mel2freq(mel)),
        "erb_rate": encoded(rate),
        "erb_width": encoded(width),
        "erb_inverse": encoded(inverse),
        "erb_inverse_width": encoded(inverse_width),
        "equal_frequency": encoded(equal_frequency),
        "equal_scale": encoded(equal_scale),
    }


def signal_utils(request):
    signal = np.asarray(request["signal"], dtype=float)
    coefficients = np.asarray(request["coefficients"], dtype=float)
    cepstrum_source = np.asarray(request["cepstrum"], dtype=float)
    equalized, level = u234.eqlz2meddis_hc_level(signal, request["out_level_db"])
    window, name = u234.taper_window(
        request["window_length"],
        request["window_kind"],
        request["taper_length"],
        request["range_sigma"],
        0,
    )
    cepstrum, minimum_phase = u234.rceps(cepstrum_source)
    frames, centers = f234.set_frame4time_sequence(
        signal, request["frame_length"], request["frame_shift"]
    )
    return {
        "equalized": encoded(equalized),
        "level": encoded(level),
        "window": encoded(window),
        "window_name": name,
        "filtered": encoded(u234.fftfilt(coefficients, signal)),
        "cepstrum": encoded(cepstrum),
        "minimum_phase": encoded(minimum_phase),
        "frames": encoded(frames),
        "centers": encoded(centers),
    }


def gammachirp_impulse(request):
    output = gc234.gammachirp(
        request["frequency"],
        request["fs"],
        request["order"],
        request["bandwidth"],
        request["chirp"],
        request["phase"],
        request["carrier"],
        request["normalization"],
    )
    return {
        "gc": encoded(output[0]),
        "length": encoded(output[1]),
        # The reference routine accidentally returns fr2fpeak's tuple here.
        "peak": encoded(output[2][0]),
        "instantaneous_frequency": encoded(output[3]),
    }


def gammachirp_response(request):
    frequencies = np.asarray(request["frequencies"], dtype=float)
    output = gc234.gammachirp_frsp(
        frequencies,
        request["fs"],
        request["order"],
        request["bandwidth"],
        request["chirp"],
        request["phase"],
        request["bins"],
    )
    phase = np.asarray(output[4]).copy()
    from scipy.special import loggamma
    phase += (
        loggamma(request["order"] + 1j * request["chirp"]).imag
        + request["chirp"] * np.log(frequencies / 1000.0)
    )[:, None]
    return {
        "amplitude": encoded(output[0]),
        "frequency": encoded(output[1]),
        "peak": encoded(output[2]),
        "group_delay": encoded(output[3]),
        "phase": encoded(phase),
    }


def asymmetric_filters(request):
    frequencies = np.asarray(request["frequencies"], dtype=float)
    bandwidth = np.asarray(request["bandwidth"], dtype=float)
    chirp = np.asarray(request["chirp"], dtype=float)
    coefficients = f234.make_asym_cmp_filters_v2(
        request["fs"], frequencies.reshape(-1, 1), bandwidth.reshape(-1, 1), chirp.reshape(-1, 1)
    )
    response = f234.asym_cmp_frsp_v2(
        frequencies, request["fs"], bandwidth, chirp, request["bins"], 4
    )
    _, status = f234.acfilterbank(coefficients, [])
    sequence = []
    for samples in request["samples"]:
        output, status = f234.acfilterbank(coefficients, status, np.asarray(samples), request["reverse"])
        sequence.append(np.asarray(output)[:, 0])
    return {
        "ap": encoded(coefficients.ap),
        "bz": encoded(coefficients.bz),
        "response": encoded(response[0]),
        "response_frequency": encoded(response[1]),
        "asymmetry": encoded(response[2]),
        "sequence": encoded(sequence),
    }


def compressed_response(request):
    frequencies = np.asarray(request["frequencies"], dtype=float)
    b1 = np.asarray(request["b1"], dtype=float)
    c1 = np.asarray(request["c1"], dtype=float)
    ratio = np.asarray(request["ratio"], dtype=float)
    b2 = np.asarray(request["b2"], dtype=float)
    c2 = np.asarray(request["c2"], dtype=float)
    output = digital_compressive_response(
        frequencies, request["fs"], request["order"], b1, c1, ratio, b2, c2,
        request["bins"],
    )
    return {
        key: encoded(value) for key, value in output.items()
    }


def frequency_conversion(request):
    arguments = (
        request["order"], request["b1"], request["c1"], request["b2"],
        request["c2"], request["ratio"],
    )
    peak, second_center = f234.fr1_to_fp2(*arguments, np.asarray([request["fr1"]]))
    peak = float(np.real(np.asarray(peak).reshape(-1)[0]))
    inverse_center, inverse_peak = f234.fp2_to_fr1(*arguments, peak)
    return {
        "peak": peak,
        "second_center": float(np.real(np.asarray(second_center).reshape(-1)[0])),
        "inverse_center": float(np.real(np.asarray(inverse_center).reshape(-1)[0])),
        "inverse_peak": float(np.real(np.asarray(inverse_peak).reshape(-1)[0])),
    }


def v234_utils(request):
    x = np.asarray(request["x"], dtype=float)
    y = np.asarray(request["y"], dtype=float)
    queries = np.asarray(request["queries"], dtype=float)
    signal = np.asarray(request["signal"], dtype=float)
    equalized, level = u234.eqlz2meddis_hc_level(
        signal, request["out_level_db"], request["input_rms1_dbspl"]
    )
    values = np.asarray(request["floor_values"], dtype=float).reshape(request["floor_shape"])
    floor = None if request["floor"] == "none" else "ZeroFloor"
    return {
        "interpolated": encoded(u234.interp1(x, y, queries, extrapolate=request["extrapolate"])),
        "equalized": encoded(equalized),
        "level": encoded(level),
        "floor": encoded(u234.eqlz_gcfb2rms1_at_0db(values, floor)),
        "spl": float(np.asarray(u234.hl2spl(np.asarray(request["hl_frequency"]), request["hl_db"])).reshape(-1)[0]),
        "cochlea": float(np.asarray(u234.hl2pin_cochlea(np.asarray(request["hl_frequency"]), request["hl_db"])).reshape(-1)[0]),
    }


def field_transfer(request):
    param = u234.param_trans_func(
        fs=request["fs"],
        n_frq_rsl=request["bins"],
        freq_calib=request["calibration"],
        type_field2eardrum=request["field"],
        type_midear2cochlea="MiddleEar",
    )
    output, _ = u234.trans_func_field2cochlea(param)
    return {
        "frequency": encoded(output.freq),
        "field": encoded(output.field2eardrum_db),
        "middle": encoded(output.midear2cochlea_db),
        "total": encoded(output.field2cochlea_db),
        "frequency_calibration": float(output.freq_calib),
        "field_at_calibration": float(output.field2eardrum_db_at_freq_calib),
        "field_compensation": float(output.field2eardrum_db_cmpnst_db),
        "middle_at_calibration": float(output.midear2cochlea_db_at_freq_calib),
        "total_at_calibration": float(output.field2cochlea_db_at_freq_calib),
    }


def modulation_filterbank(request):
    param = Param()
    param.fs = request["fs"]
    param.fc_mod_list = np.asarray(request["center_frequencies"], dtype=float)
    return encoded(f234.gcfb_v23_env_mod_fb(np.asarray(request["signal"], dtype=float), param))


def envelope_modulation_loss(request):
    frames = np.asarray(request["frames"], dtype=float)
    parameter_request = {
        "fs": request["fs"],
        "channels": int(frames.shape[0]),
        "f_range": request["f_range"],
        "hearing_loss": "NH",
    }
    param, _ = f234.set_param(make_v234_param(parameter_request))
    # The reference set_param keeps channel frequencies as a column vector;
    # flatten it for the scalar-per-channel envelope routine.
    param.fr1 = np.asarray(param.fr1).reshape(-1)
    em = Param()
    em.reduce_db = np.asarray(request["reduce_db"], dtype=float)
    em.f_cutoff = np.asarray(request["f_cutoff"], dtype=float)
    output, em = f234.gcfb_v23_env_mod_loss(frames, param, em)
    return {
        "output": encoded(output),
        "fs": float(em.fs),
        "fb_fr1": encoded(em.fb_fr1),
        "fb_reduce_db": encoded(em.fb_reduce_db),
        "fb_f_cutoff": encoded(em.fb_f_cutoff),
    }


def v234_asymmetric_io(request):
    param, response = f234.set_param(make_v234_param(request))
    pins = np.asarray(request["pins"], dtype=float)
    asymmetry, output, _ = f234.gcfb_v23_asym_func_in_out(
        param, response, request["query_frequency"], request["health"], pins
    )
    inverse = [
        f234.gcfb_v23_asym_func_in_out_inv_io_func(
            param, response, request["query_frequency"], request["health"], value
        )
        for value in output
    ]
    return {
        "asymmetry": encoded(asymmetry),
        "output": encoded(output),
        "inverse": encoded(inverse),
    }


def v234_filterbank(request):
    output = f234.gcfb_v234(np.asarray(request["signal"], dtype=float), make_v234_param(request))
    dcgc, scgc, param, response = output
    old_response = f234.cmprs_gc_frsp(
        np.asarray(response.fr1).reshape(-1), param.fs, param.n,
        np.asarray(response.b1_val).reshape(-1, 1),
        np.asarray(response.c1_val).reshape(-1, 1),
        np.asarray([param.lvl_est.frat]), np.asarray([param.lvl_est.b2]),
        (np.asarray(param.hloss.fb_compression_health).reshape(-1) * param.lvl_est.c2).reshape(-1, 1),
        2048,
    )
    old_normalization = np.asarray(old_response.norm_fct_fp2).reshape(-1)
    new_normalization = realized_frame_normalization(param, response)
    dcgc = np.asarray(dcgc) * (new_normalization / old_normalization)[:, None]
    return {
        "dcgc": encoded(dcgc),
        "scgc": encoded(scgc),
        "fr2": encoded(response.fr2),
        "ratio": encoded(response.frat_val),
        "level": encoded(response.lvl_db),
        "level_frame": encoded(getattr(response, "lvl_db_frame", [])),
        "pgc_frame": encoded(getattr(response, "pgc_frame", [])),
        "scgc_frame": encoded(getattr(response, "scgc_frame", [])),
        "asymmetry_gain": encoded(getattr(response, "asym_func_gain", [])),
        "gain": encoded(getattr(response, "gain_factor", [])),
    }


OPERATIONS = {
    "scales": scales,
    "signal_utils": signal_utils,
    "gammachirp_impulse": gammachirp_impulse,
    "gammachirp_response": gammachirp_response,
    "asymmetric_filters": asymmetric_filters,
    "compressed_response": compressed_response,
    "frequency_conversion": frequency_conversion,
    "v234_utils": v234_utils,
    "field_transfer": field_transfer,
    "modulation_filterbank": modulation_filterbank,
    "envelope_modulation_loss": envelope_modulation_loss,
    "v234_asymmetric_io": v234_asymmetric_io,
    "v234_filterbank": v234_filterbank,
}


def initialize():
    global np, u234, gc234, f234

    import numpy as np_module

    np = np_module
    # A few reference-domain warnings use an interactive confirmation.  The
    # generated strategies stay in the supported domain, but never let an
    # accidental boundary case hang a non-interactive test process.
    builtins.input = lambda _prompt="": ""
    # Plotting is not involved in the oracle operations.  Let lean Python test
    # environments omit Matplotlib even though the reference modules import it.
    try:
        import matplotlib.pyplot  # noqa: F401
    except ModuleNotFoundError:
        matplotlib = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")
        matplotlib.pyplot = pyplot
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.pyplot"] = pyplot

    u234, gc234, f234 = load_version("gcfb_v234")


def main():
    try:
        initialize()
        import scipy
        ready = {
            "ready": True,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        }
    except Exception as error:  # Dependency/import diagnostics are protocol data.
        ready = {"ready": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(ready), flush=True)
        return

    print(json.dumps(ready), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = OPERATIONS[request["op"]]
            # Several reference functions print diagnostics unconditionally;
            # keep those messages from corrupting the JSON-lines protocol.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = operation(request)
            response = {"ok": True, "result": result}
            print(json.dumps(response, allow_nan=False, separators=(",", ":")), flush=True)
        except Exception as error:
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
            print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
