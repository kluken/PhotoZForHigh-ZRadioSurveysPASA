#! /usr/bin/env python

from scripts.helperFuncs import *
from datetime import datetime
from redshift_utils import *
from sys import exit
import argparse, os, shutil
import numpy as np
import pandas as pd

# import pandas as pd


def main():
    start_time = datetime.now()
    parser = argparse.ArgumentParser(description="This script runs the ANNz algorithm, after reading, splitting and creating datasets")
    parser.add_argument("-s", "--seed", nargs=1, required=True, type=int, help="Index of random seed to use") 
    parser.add_argument("-l", "--logDisable", action='store_true', help="Disable the screen outputs") 


    args = vars(parser.parse_args())


    seed = args["seed"][0]
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
    rand_seed = seed_list[seed]
    
    if args["logDisable"]:
        glob.pars["truncateLog"] = True
        glob.pars["logLevel"] = "INFO"
        log_file = "seed_"+str(rand_seed)+".log"
        glob.pars["logFileName"] = log_file
    else:
        glob.pars["truncateLog"] = False
        glob.pars["logLevel"] = "INFO"
        glob.pars["logFileName"] = ""

    


    test_split_rate = 0.3 # 30% of total is taken for test set
    validate_split_rate = 0.3 # 30% of training set (70% of total) is taken for validation
    num_classification_bins = 30 

    # Read files
    atlas_file = "../../../Data/ATLAS_Corrected.fits"
    stripe_file = "../../../Data/Stripe_Corrected.fits"
    sdss_file = "../../../Data/RGZ_NVSS_AllWISE_SDSS_NotStripe.fits"

    des_cols = ["z", "g_corrected", "r_corrected", "i_corrected", "z_corrected",
        "W1Mag", "W2Mag", "W3Mag", "W4Mag"]
    sdss_cols = ["z", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z",
        "W1Mag", "W2Mag", "W3Mag", "W4Mag"]

    atlas_data = read_fits(atlas_file, des_cols)
    stripe_data = read_fits(stripe_file, sdss_cols)
    sdss_data = read_fits(sdss_file, sdss_cols)

    combined_data = np.vstack((atlas_data, stripe_data, sdss_data))

    x_vals = combined_data[:, 1:]
    y_vals = combined_data[:,0]

    # np.random.seed(seed)
    # x_vals_train, x_vals_test, y_vals_train, y_vals_test = split_data(x_vals, y_vals, False, test_split_rate)
    # np.random.seed(seed)
    # x_vals_train, x_vals_val, y_vals_train, y_vals_val = split_data(x_vals_train, y_vals_train, False, validate_split_rate)

    # y_vals_train_bin, y_vals_val_bin, bin_edge, bin_median\
    #         = bin_data_func(np.copy(y_vals_train), np.copy(y_vals_val), num_classification_bins)
    # y_vals_train_bin, y_vals_test_bin, bin_edge, bin_median\
    #         = bin_data_func(np.copy(y_vals_train), np.copy(y_vals_test), num_classification_bins)
    # train_numpy = np.hstack((np.expand_dims(y_vals_train_bin, axis=1), x_vals_train))
    # valid_numpy = np.hstack((np.expand_dims(y_vals_val_bin, axis=1), x_vals_val))
    # test_numpy = np.hstack((np.expand_dims(y_vals_test_bin, axis=1), x_vals_test))

    x_vals_train, x_vals_val, x_vals_test, y_vals_train_bin, y_vals_val_bin, y_vals_test_bin \
        = split_train_valid_test(x_vals, y_vals, seed)

    # Calculate the bin edges and bin medians only from the training set
    _, _, bin_edge, bin_median = bin_data_func(y_vals_train_bin, y_vals_test_bin, num_classification_bins)

    y_vals_train_bin, y_vals_val_bin, _, _\
            = bin_data_func(np.copy(y_vals_train_bin), np.copy(y_vals_val_bin), \
                num_classification_bins, bin_edge, bin_median)
    y_vals_train_bin, y_vals_test_bin, _, _\
            = bin_data_func(np.copy(y_vals_train_bin), np.copy(y_vals_test_bin), \
                num_classification_bins, bin_edge, bin_median)


    train_numpy = np.hstack((np.expand_dims(y_vals_train_bin, axis=1), x_vals_train))
    valid_numpy = np.hstack((np.expand_dims(y_vals_val_bin, axis=1), x_vals_val))
    test_numpy = np.hstack((np.expand_dims(y_vals_test_bin, axis=1), x_vals_test))


    col_names = ["#z", "mag_g", "mag_r", "mag_i", "mag_z",
        "W1Mag", "W2Mag", "W3Mag", "W4Mag"]



    
    train_file_name = "train_" + str(rand_seed) + ".csv"
    valid_file_name = "validate_" + str(rand_seed) + ".csv"
    test_file_name = "test_" + str(rand_seed) + ".csv"
    train_numpy_df = pd.DataFrame(data=train_numpy, columns=col_names)
    train_numpy_df.to_csv(train_file_name, index=False)
    valid_numpy_df = pd.DataFrame(data=valid_numpy, columns=col_names)
    valid_numpy_df.to_csv(valid_file_name, index=False)
    test_numpy_df = pd.DataFrame(data=test_numpy, columns=col_names)
    test_numpy_df.to_csv(test_file_name, index=False)

    #ANNz Initialisation
    initLogger()
    setCols()
    initROOT()


    # Running ANNz:
    log.info(whtOnBlck(" - "+time.strftime("%d/%m/%y %H:%M:%S")+" - starting ANNZ"))
    
    # ANNz Settings:
    glob.annz["nMLMs"]      = 100
    glob.annz["zTrg"]       = "z"
    glob.annz["minValZ"]    = 0.0
    glob.annz["maxValZ"]    = 7
    glob.annz["nErrKNN"]    = 100
    glob.annz["sampleFrac_errKNN"] = 1

    # Input Settings:
    glob.annz["inDirName"]    = "." #Location of input data
    glob.annz["splitTypeTrain"]  = "train_"+str(rand_seed)+".csv" # Training Data Set
    glob.annz["splitTypeTest"]   = "validate_"+str(rand_seed)+".csv" # Validation Set
    glob.annz["inAsciiFiles"]   = "test_"+str(rand_seed)+".csv" # Test Set
    glob.annz["inAsciiVars"]  = "D:z;F:mag_g;F:mag_r;F:mag_i;F:mag_z;D:W1mag;D:W2mag;D:W3mag;D:W4mag" # Input variables

    # Output Settings:
    glob.annz["outDirName"] = "Results" # Location of output results (models, results, etc) - Will be created
    glob.annz["evalDirPostfix"] = "seed_" + str(rand_seed) # Place to store the final results

    # General ANNz Opts:
    glob.annz["doGenInputTrees"] = False
    glob.annz["doTrain"] = False
    glob.annz["doOptim"] = False
    glob.annz["doVerif"] = False
    glob.annz["doEval"] = False
    glob.annz["doRegression"] = True
    glob.annz["doRandomReg"] = True
    glob.annz["doClassification"] = False
    glob.annz["doBinnedCls"] = False
    glob.annz["isBatch"] = True
    glob.annz["doPlots"] = False
    glob.annz["printPlotExtension"] = "pdf"

    # If you want to weight the input trees by the output data
    useWgtKNN = False
    if useWgtKNN:
        glob.annz["useWgtKNN"]             = True
        glob.annz["minNobjInVol_wgtKNN"]   = 5
        glob.annz["inAsciiFiles_wgtKNN"]   = "test_"+str(rand_seed)+".csv"
        glob.annz["inAsciiVars_wgtKNN"]    = "D:z;F:mag_g;F:mag_r;F:mag_i;F:mag_z;D:W1mag;D:W2mag;D:W3mag;D:W4mag"    
        glob.annz["weightVarNames_wgtKNN"] = "mag_g;mag_r;mag_i;mag_z;W1mag;W2mag;W3mag;W4mag"
        glob.annz["sampleFracInp_wgtKNN"]  = 1                                         # fraction of dataset to use (positive number, smaller or equal to 1)
        glob.annz["sampleFracRef_wgtKNN"]  = 1                                          # fraction of dataset to use (positive number, smaller or equal to 1)
        glob.annz["outAsciiVars_wgtKNN"]   = ""                        # write out two additional variables to the output file
        glob.annz["weightRef_wgtKNN"]      = "" # down-weight objects with high MAGERR_R
        glob.annz["cutRef_wgtKNN"]         = ""                                # only use objects which have small MAGERR_U
        glob.annz["doWidthRescale_wgtKNN"] = True
        glob.annz["trainTestTogether_wgtKNN"] = False

    # Run first stage - Generate input trees
    glob.annz["doGenInputTrees"] = True

    runANNZ()

    ######################################################################################
    # Training

    glob.annz["doGenInputTrees"] = False
    glob.annz["doTrain"] = True
    glob.annz["trainIndex"] = -1

    for nMLMnow in range(glob.annz["nMLMs"]): 
        if nMLMnow == 17:
            continue
        start_time = datetime.now()
        glob.annz["nMLMnow"] = nMLMnow
        if glob.annz["trainIndex"] >= 0 and glob.annz["trainIndex"] != nMLMnow: continue

        glob.annz["inputVariables"] = "W1mag;W2mag;W3mag;W4mag;mag_g;mag_r;mag_i;mag_z"
        glob.annz["inputVarErrors"] = "" # Or add errors if wanted - finding by kNN seems to work

        glob.annz["userMLMopts"] = "" # Use a mix of BDT and ANN
        glob.annz["rndOptTypes"] = "ANN_BDT"

        # --------------------------------------------------------------------------------------------------
        # bias-correction procedure on MLMs -
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #   - doBiasCorMLM      - whether or not to perform the correction for MLMs (during training)
        #   - biasCorMLMopt     - MLM configuration options for the bias-correction for MLMs - simple structures are recommended !!!
        #                         - can take the same format as userMLMopts (e.g., [biasCorMLMopt="ANNZ_MLM=BDT:VarTransform=N:NTrees=100"])
        #                         - can be empty (then the job options will be automatically generated, same as is setting [userMLMopts=""])
        #                         - can be set as [biasCorMLMopt="same"], then the same configuration options as for the nominal MLM
        #                           for which the bias-correction is applied are used
        #                       - simple MLMs are recommended, e.g.: 
        #                         - BDT with around 50-100 trees:
        #                           "ANNZ_MLM=BDT:VarTransform=N:NTrees=100:BoostType=AdaBoost"
        #                         - ANN with a simple layer structure, not too many NCycles etc.:
        #                           "ANNZ_MLM=ANN::HiddenLayers=N,N+5:VarTransform=N,P:TrainingMethod=BFGS:TestRate=5:NCycles=500:UseRegulator=True"
        #   - biasCorMLMwithInp - add the nominal MLM as an input variable for the new MLM of the bias-correction (not necessary, as it
        #                         may just add noise)
        #   - alwaysKeepBiasCor - whether or not to not check the KS-test and N_poiss metrics for improvement in order to
        #                         possibly reject the bias correction (check performed if [alwaysKeepBiasCor] is set to True)
        # - example use:
        # -----------------------------------------------------------------------------------------------------------
        doBiasCorMLM = True
        if doBiasCorMLM:
            glob.annz["doBiasCorMLM"]      = True
            glob.annz["biasCorMLMwithInp"] = False
            glob.annz["alwaysKeepBiasCor"] = False
            # as an example, a couple of choices of MLM options (this shouldn't matter much though...)
            if nMLMnow % 2 == 0:
                glob.annz["biasCorMLMopt"]   = "ANNZ_MLM=BDT:VarTransform=N:NTrees=50:BoostType=AdaBoost"
            else:
                glob.annz["biasCorMLMopt"]   = "ANNZ_MLM=ANN:HiddenLayers=N+5:VarTransform=N,P:TrainingMethod=BFGS:NCycles=500:UseRegulator=True"

        # run ANNZ with the current settings
        runANNZ()

    ######################################################################################
    # Optimisation

    glob.annz["doTrain"] = False
    glob.annz["doOptim"] = True


    # --------------------------------------------------------------------------------------------------
    # nPDFs - number of PDFs (up to three PDF types are implemented for randomized regression):
    #   - PDFs are selected by choosing the weighting scheme which is "most compatible" with the true value.
    #     This is determined in several ways (generating alternative PDFs), using cumulative distributions; for the first/second
    #     PDFs (PDF_0 and PDF_1 in the output), the cumulative distribution is based on the "truth" (the target variable, zTrg).
    #     For the third PDF (PDF_2 in the output), the cumulative distribution is based on the "best" MLM.
    #     For the former, a set of templates, derived from zTrg is used to fit the dataset. For the later,
    #     a flat distribution of the cumulator serves as the baseline.
    #   - nominally, only PDF_0 should be used. PDF_1/PDF_2 are depracated, and are not guaranteed to
    #     be supported in the future. They may currently still be generated by setting addOldStylePDFs to True
    # --------------------------------------------------------------------------------------------------
    glob.annz["nPDFs"] = 1

    speedUpMCMC = False
    if speedUpMCMC:
        # max_optimObj_PDF - can be set to some number (smaller than the size of the training sample). the
        # result will be to limit the number of objects used in each step of the MCMC used to derive the PDF
        glob.annz["max_optimObj_PDF"] = 300
        # nOptimLoops - set the maximal number of steps taken by the MCMC
        # note that the MCMC will likely end before `nOptimLoops` steps in any case. this will happen
        # after a pre-set number of steps, during which the solution does not improve.
        glob.annz["nOptimLoops"] = 5000

    glob.annz["nPDFbins"]    = 120 # When generating PDFs, use this number of bins

    glob.annz["MLMsToStore"] = "" # Don't actually store any MLMs

    glob.annz["addOutputVars"] = "" #When saving results, add this column

    # --------------------------------------------------------------------------------------------------
    # max_sigma68_PDF, max_bias_PDF, max_frac68_PDF
    #   - if max_sigma68_PDF, max_bias_PDF are positive, they put thresholds on the maximal value of
    #     the scatter (max_sigma68_PDF), bias (max_bias_PDF) or outlier-fraction (max_frac68_PDF) of an
    #     MLM which may be included in the PDF created in randomized regression
    # --------------------------------------------------------------------------------------------------
    glob.annz["max_sigma68_PDF"] = 0.15
    glob.annz["max_bias_PDF"]    = 0.1
    glob.annz["max_frac68_PDF"]  = 0.15

    # --------------------------------------------------------------------------------------------------
    # set the minimal acceptable number of MLMs used to generate PDFs
    # (in practice, it's recommended to keep this number high...)
    # --------------------------------------------------------------------------------------------------
    glob.annz["minAcptMLMsForPDFs"]  = 10

    runANNZ()


    ######################################################################################
    # Evaluation
    glob.annz["doOptim"] = False
    glob.annz["doEval"] = True



    # run ANNZ with the current settings
    runANNZ()

    log.info(whtOnBlck(" - "+time.strftime("%d/%m/%y %H:%M:%S")+" - finished running ANNZ !"))

    ######################################################################################
    # Tidy up!
    old_results_file = "output/Results/regres/eval_seed_" + str(rand_seed) + "/ANNZ_randomReg_0000.csv"
    new_results_file = "./predictions.csv"
    shutil.move(old_results_file, new_results_file) 

    predictions = np.loadtxt(new_results_file, delimiter=',', skiprows=1) 
    re_bin_spec, re_bin_photo, _, _ = bin_data_func(predictions[:,0], predictions[:,1], num_classification_bins, bin_edges=bin_edge, bin_medians=bin_median)
    hex_plot_filename = "final_plot.pdf"
    plot_hex(re_bin_spec, re_bin_photo, hex_plot_filename) 
    num_train = train_numpy.shape[0]
    num_valid = valid_numpy.shape[0]
    num_test = test_numpy.shape[0]
    out_rate = outlier_rate(norm_residual(re_bin_spec, re_bin_photo))
    time_taken = datetime.now() - start_time
    stats = {
        "num_train": num_train,
        "num_validate": num_valid,
        "num_test": num_test,
        "slurm_seed": seed,
        "split_seed": rand_seed,
        "outlier_rate": out_rate,
        "time_taken": time_taken
    }
    stats_filename = "stats.csv"
    stats_df = pd.DataFrame(data=stats, index=[0])
    stats_df.to_csv(stats_filename)

    out_pred = {
        "specz_original":y_vals_test,
        "specz_bin": y_vals_test_bin,
        "photoz_predict": predictions[:,1],
        "photoz_rebin": re_bin_photo
    }
    out_pred_filename = "binned_predictions.csv"
    out_pred_df = pd.DataFrame(data=out_pred)
    out_pred_df.to_csv(out_pred_filename)

    bin_edges = {
        "bin_edges":bin_edge
    }
    bin_edges_filename = "bin_edges.csv"
    bin_edges_df = pd.DataFrame(data=bin_edges)
    bin_edges_df.to_csv(bin_edges_filename)

    bin_medians = {
        "bin_median":bin_median
    }
    bin_medians_filename = "bin_medians.csv"
    bin_medians_df = pd.DataFrame(data=bin_medians)
    bin_medians_df.to_csv(bin_medians_filename)

    shutil.rmtree("output/Results", ignore_errors=True)

if __name__ == "__main__":
	main()
