INSERT INTO
    labeled_spectra_database (
        ds,
        raw_wavenumber,
        raw_spectrum,
        pre_spectrum,
        wave_range,
        smooth_method,
        smooth_args,
        baseline_method,
        baseline_args,
        domain
    )
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);