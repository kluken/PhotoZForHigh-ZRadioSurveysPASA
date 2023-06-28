import shutil, os

for seed in range(100):
    print("Copying. Seed: " + str(seed))
    folder_name = "seed_" + str(seed) + "/"    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    shutil.copy("ANNz_Python.simg", folder_name)
    shutil.copy("annz.py", folder_name)
    shutil.copy("redshift_utils.py", folder_name)
    shutil.copy("ATLAS_Corrected.fits", folder_name)
    shutil.copy("Stripe_Corrected.fits", folder_name)
    shutil.copy("RGZ_NVSS_AllWISE_SDSS_NotStripe_1cut.fits", folder_name)
