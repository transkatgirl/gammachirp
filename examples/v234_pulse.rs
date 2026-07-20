use gammachirp_rs::gcfb_v234::{GcParam, GcfbStream};

fn main() -> gammachirp_rs::Result<()> {
    let fs = 48_000;
    let period = fs / 100;
    let mut pulse_train = vec![0.0; period * 10];
    for sample in (0..pulse_train.len()).step_by(period) {
        pulse_train[sample] = 1.0;
    }

    let mut filterbank = GcfbStream::new(GcParam {
        out_mid_crct: "No".into(),
        ..GcParam::default()
    })?;
    let channels = filterbank.gc_param().num_ch;
    let mut output_samples = 0;
    for sample in pulse_train {
        if filterbank.process_sample(sample)?.event.is_some() {
            output_samples += 1;
        }
    }
    output_samples += filterbank.finish()?.len();
    println!("{channels} channels × {output_samples} streaming outputs");
    Ok(())
}
