from datetime import datetime

from matplotlib.pyplot import legend
from redshift_utils import *
import subprocess,argparse, os
import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from sklearn.mixture import GaussianMixture
from collections import OrderedDict
import subprocess

def main():
    parser = argparse.ArgumentParser(description="This script runs the GPz algorithm, after reading, splitting and creating datasets")
    parser.add_argument("-s", "--seed", nargs=1, required=True, type=int, help="Index of random seed to use") 
    args = vars(parser.parse_args())
    seed_arg = args["seed"][0]
    seed_list = [
            28038,243145,205639,137877,136035,269736,27000,219084,63293,29586,165398,
            306500,231143,239118,225401,246949,161234,40248,263814,141492,157190,116489,
            57349,291151,245536,202276,126422,258477,171351,139302,141515,71389,28945,
            174227,278938,20048,269639,260007,86946,198443,51908,238160,220075,111377,
            21337,304953,140016,280582,212974,244536,238729,61147,114324,146624,156385,
            13761,171709,48471,233538,214585,289820,233973,115184,303951,129072,102360,
            284482,116383,23983,147515,249970,59524,145436,40816,215663,149446,103734,
            71285,177342,210428,295405,137335,50481,261593,197846,219994,30544,98132,
            241225,261461,136727,252823,264425,121729,282142,90580,75259,214412,200044,43904
            ]
    seed = seed_list[seed_arg]
    test_split_rate = 0.3 # 30% of total is taken for test set

    # Default GPz options:
    gpz_opts = OrderedDict()
    gpz_opts["VERBOSE"] = 1
    gpz_opts["N_THREAD"] = 4
    gpz_opts["TRAINING_CATALOG"] = "dummy"
    gpz_opts["PREDICTION_CATALOG"] = "dummy"
    gpz_opts["BANDS"] = "[mag]"
    gpz_opts["FLUX_COLUMN_PREFIX"] = "mag_"
    gpz_opts["ERROR_COLUMN_PREFIX"] = "magerr_"
    gpz_opts["OUTPUT_COLUMN"] = "z_spec"
    gpz_opts["WEIGHT_COLUMN"] = ""
    gpz_opts["WEIGHTING_SCHEME"] = "uniform"
    gpz_opts["OUTPUT_MIN"] = 0
    gpz_opts["OUTPUT_MAX"] = 7
    gpz_opts["USE_ERRORS"] = 1
    gpz_opts["TRANSFORM_INPUTS"] = "no"   
    gpz_opts["NORMALIZATION_SCHEME"] = "whiten"
    gpz_opts["VALID_SAMPLE_METHOD"] = "random"
    gpz_opts["TRAIN_VALID_RATIO"] = 0.7
    gpz_opts["VALID_SAMPLE_SEED"] = 42
    gpz_opts["OUTPUT_CATALOG"] = "dummy"
    gpz_opts["MODEL_FILE"] = "dummy"
    gpz_opts["SAVE_MODEL"] = 1
    gpz_opts["REUSE_MODEL"] = 0
    gpz_opts["USE_MODEL_AS_HINT"] = 0
    gpz_opts["PREDICT_ERROR"] = 1
    gpz_opts["NUM_BF"] = 50
    gpz_opts["COVARIANCE"] = "gpvd"
    gpz_opts["PRIOR_MEAN"] = "constant"
    gpz_opts["OUTPUT_ERROR_TYPE"] = "input_dependent"
    gpz_opts["BF_POSITION_SEED"] = 55
    gpz_opts["FUZZING"] = 0
    gpz_opts["FUZZING_SEED"] = 42
    gpz_opts["MAX_ITER"] = 500
    gpz_opts["TOLERANCE"] = 1e-9
    gpz_opts["GRAD_TOLERANCE"] = 1e-5


    atlas_file = "../../../Data/ATLAS_Corrected.fits"
    stripe_file = "../../../Data/Stripe_Corrected.fits"
    sdss_file = "../../../Data/RGZ_NVSS_AllWISE_SDSS_NotStripe.fits"

    des_cols = ["z", "g_corrected", "r_corrected", "i_corrected", "z_corrected",
        "g_corrected_err", "r_corrected_err", "i_corrected_err", "z_corrected_err",
        "W1Mag", "W2Mag", "W3Mag", "W4Mag",
        "e_W1Mag", "e_W2Mag", "e_W3Mag", "e_W4Mag"]
    sdss_cols = ["z", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z",
        "modelMagErr_g", "modelMagErr_r", "modelMagErr_i", "modelMagErr_z",
        "W1Mag", "W2Mag", "W3Mag", "W4Mag",
        "e_W1Mag", "e_W2Mag", "e_W3Mag", "e_W4Mag"]

    atlas_data = read_fits(atlas_file, des_cols)
    stripe_data = read_fits(stripe_file, sdss_cols)
    sdss_data = read_fits(sdss_file, sdss_cols)

    combined_data = np.vstack((atlas_data, stripe_data, sdss_data))

    combined_data[np.where(combined_data[:,-4] == -99), -4] = np.nanmax(combined_data[:,-4])
    combined_data[np.where(combined_data[:,-3] == -99), -3] = np.nanmax(combined_data[:,-3])
    combined_data[np.where(combined_data[:,-2] == -99), -2] = np.nanmax(combined_data[:,-2])
    combined_data[np.where(combined_data[:,-1] == -99), -1] = np.max(combined_data[:,-1])

    combined_data[np.where(np.isnan(combined_data[:,-4])), -4] = np.nanmax(combined_data[:,-4])
    combined_data[np.where(np.isnan(combined_data[:,-3])), -3] = np.nanmax(combined_data[:,-3])
    combined_data[np.where(np.isnan(combined_data[:,-2])), -2] = np.nanmax(combined_data[:,-2])
    combined_data[np.where(np.isnan(combined_data[:,-1])), -1] = np.nanmax(combined_data[:,-1])

    x_vals = combined_data[:, 1:]
    y_vals = combined_data[:,0]


    gmm_comp = 30

    x_vals_train, x_vals_test, y_vals_train, y_vals_test = split_data(np.copy(x_vals), np.copy(y_vals), seed, test_split_rate)


    # for gmm_comp in gmm_components_list:
    gmm_model = GaussianMixture(n_components=gmm_comp,
                        max_iter=500).fit(x_vals_train)

    clusters_train = gmm_model.predict(x_vals_train)
    clusters_test = gmm_model.predict(x_vals_test)
    

    for component in np.arange(gmm_model.n_components):
        test_file_name = "pred_comp_" + str(component) + ".txt"
        if not os.path.isfile(test_file_name):
            print("\n#############################\n Component: ", str(component))
            comp_y_train = y_vals_train[np.where(clusters_train == component)]
            comp_x_train = x_vals_train[np.where(clusters_train == component)]
            comp_y_test = y_vals_test[np.where(clusters_test == component)]
            comp_x_test = x_vals_test[np.where(clusters_test == component)]
            train_data = np.hstack((np.expand_dims(comp_y_train, axis=1), comp_x_train))
            test_data = np.hstack((np.expand_dims(comp_y_test, axis=1), comp_x_test))


            col_names = ["z_spec", "mag_g", "mag_r", "mag_i", "mag_z",
                "magerr_g", "magerr_r", "magerr_i", "magerr_z",
                "mag_w1", "mag_w2", "mag_w3", "mag_w4",
                "magerr_w1", "magerr_w2", "magerr_w3", "magerr_w4"]
            train_table = Table(train_data, names=col_names)
            test_table = Table(test_data, names=col_names)
            
            train_file = "train_comp_" + str(component) + ".cat"
            test_file = "test_comp_" + str(component) + ".cat"
            
            train_table.write(train_file, format='ascii.commented_header', overwrite=True)
            test_table.write(test_file, format='ascii.commented_header', overwrite=True)

            gpz_opts["TRAINING_CATALOG"] = "train_comp_" + str(component) + ".cat"
            gpz_opts["PREDICTION_CATALOG"] = "test_comp_" + str(component) + ".cat"
            gpz_opts["OUTPUT_CATALOG"] = "pred_comp_" + str(component) + ".txt"
            gpz_opts["MODEL_FILE"] = "model_comp_" + str(component) + ".cat"

            gpz_file = "gpz_params_comp_" + str(component) + ".param"

            out_file = open(gpz_file, "w")
            for param in gpz_opts:
                out_file.write('{0:30s} = {1}\n'.format(param, gpz_opts[param]))
            out_file.close()

            cmd = ['gpz++', gpz_file]
            subprocess.Popen(cmd).wait()
            print("\n#############################\n")
            
        
        
    pred_file_name = "pred_comp_0.txt"
    train_file_name = "train_comp_0.cat"
    test_file_name = "test_comp_0.cat"
    pred_table_stack = Table.read(pred_file_name, format="ascii.commented_header", header_start=10)
    test_table_stack = Table.read(test_file_name, format="ascii.commented_header")
    train_table_stack = Table.read(train_file_name, format="ascii.commented_header")

    test_table_stack["prediction"] = pred_table_stack["value"]
    test_table_stack["prediction_uncert"] = pred_table_stack["uncertainty"]
    test_table_stack["comp_no"] = 0
    train_table_stack["comp_no"] = 0


    for component in np.arange(1,gmm_model.n_components):
        pred_file_name = "pred_comp_" + str(component) + ".txt"
        test_file_name = "test_comp_" + str(component) + ".cat"
        train_file_name = "train_comp_" + str(component) + ".cat"
        pred_table = Table.read(pred_file_name, format="ascii.commented_header", header_start=10)
        test_table = Table.read(test_file_name, format="ascii.commented_header")
        train_table = Table.read(train_file_name, format="ascii.commented_header")

        test_table["prediction"] = pred_table["value"]
        test_table["prediction_uncert"] = pred_table["uncertainty"]
        test_table["comp_no"] = component
        train_table["comp_no"] = component

        test_table_stack = vstack([test_table_stack, test_table])
        train_table_stack = vstack([train_table_stack, train_table])


    test_table_stack.write("predictions_seed_"+str(seed)+"_comp_" + str(gmm_comp) + ".csv", format="csv", overwrite=True)
    train_table_stack.write("train_seed_"+str(seed)+"_comp_" + str(gmm_comp) + ".csv", format="csv", overwrite=True)

    hex_plot_filename = "final_plot_seed_"+str(seed)+"_comp_" + str(gmm_comp) + ".pdf"
    plot_hex(test_table_stack["z_spec"], test_table_stack["prediction"], hex_plot_filename) 
    out_rate = outlier_rate(norm_residual(test_table_stack["z_spec"], test_table_stack["prediction"]))

    print("Outlier rate: " + str(np.round(out_rate, 2)))




if __name__ == "__main__":
	main()
