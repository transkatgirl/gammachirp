use gammachirpy::gcfb_v234::{GcParam, gcfb_v234};

fn main() -> gammachirpy::Result<()> {
    let fs = 48_000;
    let period = fs / 100;
    let mut pulse_train = vec![0.0; period * 10];
    for sample in (0..pulse_train.len()).step_by(period) {
        pulse_train[sample] = 1.0;
    }

    let parameters = GcParam {
        out_mid_crct: "No".into(),
        ..GcParam::default()
    };
    let output = gcfb_v234(&pulse_train, parameters)?;
    println!(
        "{} channels × {} frames",
        output.dcgc_out.nrows(),
        output.dcgc_out.ncols()
    );
    Ok(())
}
