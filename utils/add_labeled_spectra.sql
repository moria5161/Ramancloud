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
        domain,
        data_hash


    )
VALUES
    (
        "{}",
        "{}", 
        "{}", 
        "{}", 
        "{}", 
        "{}", 
        "{}", 
        "{}", 
        "{}",
        "{}",
        SHA1(raw_spectrum)  
    );