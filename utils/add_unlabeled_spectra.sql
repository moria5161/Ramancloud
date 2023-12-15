INSERT INTO
    unlabeled_spectra_database (
        ds,
        raw_wavenumber,
        raw_spectrum,
        data_hash
    )
VALUES
    (
        "{}",
        "{}", 
        "{}", 
        SHA1(raw_spectrum)  
    );