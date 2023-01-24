config file includes: 
1) adversarial examples generation; 2) adversarial examples detection; 
3) victim and threat model training; and 4) Pixel/Freq-VAE training.


1) adversarial examples generation: 
        (config_adversary.json, run-generation.py)

2) adversarial examples detection: 
        (config_lid.json run-detection.py --method lid)
        (config_md.json run-detection.py --method md)
        (config_sid.json run-detection.py --method sid)
        (config_frd.json run-frd-detection.py)
        (config_prd.json run-prd-detection.py)

3) victim and threat model training: 
        (config.json train_pipeline.py)

4) PixelVAE training:
        (config_data_reconstruction.json spat_train_pipeline.py)
   FreqVAE training:
        (config_data_reconstruction.json freq_train_pipeline.py)