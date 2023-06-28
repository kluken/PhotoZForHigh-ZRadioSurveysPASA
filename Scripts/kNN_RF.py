import numpy as np
import pickle
from redshift_utils import *
from tqdm import tqdm, trange
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error


# Read files

atlas_file = "../../Data/ATLAS_Corrected.fits"
stripe_file = "../../Data/Stripe_Corrected.fits"
sdss_file = "../../Data/RGZ_NVSS_AllWISE_SDSS_NotStripe.fits"

des_cols = ["z", "g_corrected", "r_corrected", "i_corrected", "z_corrected",
    "W1Mag", "W2Mag", "W3Mag", "W4Mag"]
sdss_cols = ["z", "modelMag_g", "modelMag_r", "modelMag_i", "modelMag_z",
    "W1Mag", "W2Mag", "W3Mag", "W4Mag"]

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


atlas_data = read_fits(atlas_file, des_cols)
stripe_data = read_fits(stripe_file, sdss_cols)
sdss_data = read_fits(sdss_file, sdss_cols)

combined_data = np.vstack((atlas_data, stripe_data, sdss_data))

x_vals = combined_data[:, 1:]
y_vals = combined_data[:,0]

overall_start = datetime.now()
# Regression tests
num_repitions = 100
## Initial run - finds value of "k" to use, and generates plots. 
x_vals_train, x_vals_test, y_vals_train, y_vals_test = split_data(x_vals, y_vals, np.random.default_rng(42).integers(314159))

folder_name = "kNN_Regress"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
os.chdir(folder_name)

## Find the best value of k to use
k_range = range(3, 31, 2)
k_fold_val = 5
random_seed_kfold = 42
out_rates = []
for k_val in tqdm(k_range):
    out_rates.append(kNN_cross_val(k_val, k_fold_val, random_seed_kfold, x_vals_train, y_vals_train))
best_k = k_range[np.argmin(out_rates)]
print("Outlier rates for the cross-validation")
print(out_rates)

## Use best k for rest. 
rand_gen = np.random.default_rng(42)
out_rates = []
out_two_sigma = []
sigma_arr = []
nmad_arr = []
mse_arr = []
for i in trange(len(seed_list)):
    seed = seed_list[i]
    # rand_num = rand_gen.integers(314159)
    folder_name = f"seed_{seed}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    os.chdir(folder_name)
    start_time = datetime.now()
    x_vals_train, x_vals_test, y_vals_train, y_vals_test = split_data(x_vals, y_vals, seed)
    x_vals_train, x_vals_test, _, _ = norm_x_vals(x_vals_train, x_vals_test)
    pred, r2, model = kNN(best_k, x_vals_train, x_vals_test, y_vals_train, y_vals_test)
    outlier = outlier_rate(norm_residual(y_vals_test, pred))
    out_rates.append(outlier)
    sigma = norm_std_dev(norm_residual(y_vals_test, pred))
    sigma_arr.append(sigma)
    outlier_sigma = outlier_rate(norm_residual(y_vals_test, pred), out_frac=(2*sigma))
    out_two_sigma.append(outlier_sigma)
    nmad = norm_mad(norm_residual(y_vals_test, pred))
    nmad_arr.append(nmad)
    mse_arr.append(mean_squared_error(y_vals_test, pred))
    end_time = datetime.now() - start_time
    if i == 0:
        plot_hex(y_vals_test, pred, file_name="knn_regression.pdf")
    prediction_filename = "predictions.csv"
    df = pd.DataFrame({"True_z" : y_vals_test, "Pred_z" : pred})
    df.to_csv(prediction_filename, index=False)
    model_filename = "model.pickle"
    with open(model_filename,  'wb') as pickle_file:
        pickle.dump(model, pickle_file)   
    stats_filename = "stats.csv"
    df = {
        "best_k":best_k,
        "num_train":y_vals_train.shape[0],
        "num_test":y_vals_test.shape[0],
        "split_seed":seed,
        "r2":r2,
        "outlier":outlier,
        "outlier_two_sigma":outlier_sigma,
        "sigma":sigma,
        "nmad":nmad,
        "time_taken":end_time
    }
    df = pd.DataFrame(data=df, index=[0])
    df.to_csv(stats_filename)
    os.chdir("../")

overall_time = datetime.now() - overall_start
print("##################")
print("kNN Regression")
print("#####\nOutliers")
print(np.mean(out_rates), (np.std(out_rates)/np.sqrt(num_repitions)), len(out_rates))
print("#####\nOutliers 2 sigma")
print(np.mean(out_two_sigma), (np.std(out_two_sigma)/np.sqrt(num_repitions)), len(out_two_sigma))
print("#####\nSigma")
print(np.mean(sigma_arr), (np.std(sigma_arr)/np.sqrt(len(sigma_arr))), len(sigma_arr))
print("#####\nNMAD")
print(np.mean(nmad_arr), (np.std(nmad_arr)/np.sqrt(len(nmad_arr))), len(nmad_arr))
print("#####\nMSE")
print(np.mean(mse_arr), (np.std(mse_arr)/np.sqrt(len(mse_arr))), len(mse_arr))
# Above is the standard error of the sample mean 
# (standard deviation divided by the square root of the number of runs)
knn_stats = {
    "best_k":best_k,
    "Outlier_rates":out_rates,
    "Mean_Out_Rate":np.mean(out_rates),
    "Std_Error_Out_Rate":(np.std(out_rates)/np.sqrt(num_repitions)),
    "Mean_Sigma_Out_Rate":np.mean(out_two_sigma),
    "Std_Error_Sigma_Out_Rate":(np.std(out_two_sigma)/np.sqrt(num_repitions)),
    "Mean_Sigma":np.mean(sigma_arr),
    "Std_Error_Sigma":(np.std(sigma_arr)/np.sqrt(num_repitions)),
    "Mean_NMAD":np.mean(nmad_arr),
    "Std_Error_NMAD":(np.std(nmad_arr)/np.sqrt(num_repitions)),
    "Mean_MSE":np.mean(mse_arr),
    "Std_Error_MSE":(np.std(mse_arr)/np.sqrt(num_repitions)),
    "Overall_time":overall_time
}
df = pd.DataFrame(data=knn_stats)
df.to_csv("knn_results.csv")
print("##################")
os.chdir("../")
folder_name = "rf_Regress"

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
os.chdir(folder_name)

# Regression tests
## Initial run - finds value of "k" to use, and generates plots. 
overall_start = datetime.now()
x_vals_train, x_vals_test, y_vals_train, y_vals_test = split_data(x_vals, y_vals, np.random.default_rng(rand_gen.integers(314159)))
# x_vals_train, x_vals_test, _, _ = norm_x_vals(x_vals_train, x_vals_test)

## Find the best value of k to use
tree_range = range(2, 61, 1)
k_fold_val = 5
random_seed_kfold = 42
out_rates = []
out_two_sigma = []
sigma_arr = []
nmad_arr = []
mse_arr = []
for tree_val in tqdm(tree_range):
    out_rates.append(rf_cross_val(tree_val, k_fold_val, x_vals_train, y_vals_train, random_seed_kfold))
best_tree = tree_range[np.argmin(out_rates)]
print("Outlier rates for the cross-validation")
print(out_rates)

## Use best k for rest. 
rand_gen = np.random.default_rng(42)
out_rates = []
for i in trange(num_repitions):
    seed = seed_list[i]
    start_time = datetime.now()
    # rand_num = rand_gen.integers(314159)
    rf_rand = rand_gen.integers(314159)
    folder_name = f"seed_{seed}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    os.chdir(folder_name)
    x_vals_train, x_vals_test, y_vals_train, y_vals_test = split_data(x_vals, y_vals, seed)
    x_vals_train, x_vals_test, _, _ = norm_x_vals(x_vals_train, x_vals_test)
    pred, r2, model = rf(best_tree, x_vals_train, x_vals_test, y_vals_train, y_vals_test, rf_rand)
    outlier = outlier_rate(norm_residual(y_vals_test, pred))
    out_rates.append(outlier)
    sigma = norm_std_dev(norm_residual(y_vals_test, pred))
    sigma_arr.append(sigma)
    outlier_sigma = outlier_rate(norm_residual(y_vals_test, pred), out_frac=(2*sigma))
    out_two_sigma.append(outlier_sigma)
    nmad = norm_mad(norm_residual(y_vals_test, pred))
    nmad_arr.append(nmad)
    mse_arr.append(mean_squared_error(y_vals_test, pred))
    end_time = datetime.now() - start_time
    if i == 0:
        plot_hex(y_vals_test, pred, file_name="rf_regression.pdf")
    prediction_filename =  "predictions.csv"
    df = pd.DataFrame({"True_z" : y_vals_test, "Pred_z" : pred})
    df.to_csv("prediction_filename.csv", index=False)
    model_filename =  "model.pickle"
    with open(model_filename,  'wb') as pickle_file:
        pickle.dump(model, pickle_file)   
    stats_filename =  "stats.csv"
    df = {
        "best_tree":best_tree,
        "num_train":y_vals_train.shape[0],
        "num_test":y_vals_test.shape[0],
        "split_seed":seed,
        "rf_seed":rf_rand,
        "r2":r2,
        "outlier":outlier,
        "outlier_two_sigma":outlier_sigma,
        "sigma":sigma,
        "nmad":nmad,
        "time_taken":end_time
    }
    df = pd.DataFrame(data=df, index=[0])
    prediction_filename = "predictions.csv"
    df = pd.DataFrame({"True_z" : y_vals_test, "Pred_z" : pred})
    df.to_csv(prediction_filename, index=False)
    df.to_csv(stats_filename)
    os.chdir("../")

overall_time = datetime.now() - overall_start
print("##################")
print("RF Regression")
print(out_rates)
print(np.mean(out_rates), (np.std(out_rates)/np.sqrt(num_repitions)), len(out_rates))
print("#####\nOutliers 2 sigma")
print(np.mean(out_two_sigma), (np.std(out_two_sigma)/np.sqrt(num_repitions)), len(out_two_sigma))
print("#####\nSigma")
print(np.mean(sigma_arr), (np.std(sigma_arr)/np.sqrt(len(sigma_arr))), len(sigma_arr))
print("#####\nNMAD")
print(np.mean(nmad_arr), (np.std(nmad_arr)/np.sqrt(len(nmad_arr))), len(nmad_arr))
print("#####\nMSE")
print(np.mean(mse_arr), (np.std(mse_arr)/np.sqrt(len(mse_arr))), len(mse_arr))
rf_stats = {
    "best_tree":best_tree,
    "Outlier_rates":out_rates,
    "Mean_Out_Rate":np.mean(out_rates),
    "Std_Error_Out_Rate":(np.std(out_rates)/np.sqrt(num_repitions)),
    "Mean_Sigma_Out_Rate":np.mean(out_two_sigma),
    "Std_Error_Sigma_Out_Rate":(np.std(out_two_sigma)/np.sqrt(num_repitions)),
    "Mean_Sigma":np.mean(sigma_arr),
    "Std_Error_Sigma":(np.std(sigma_arr)/np.sqrt(num_repitions)),
    "Mean_NMAD":np.mean(nmad_arr),
    "Std_Error_NMAD":(np.std(nmad_arr)/np.sqrt(num_repitions)),
    "Mean_MSE":np.mean(mse_arr),
    "Std_Error_MSE":(np.std(mse_arr)/np.sqrt(num_repitions)),
    "Overall_time":overall_time
}
df = pd.DataFrame(data=rf_stats)
df.to_csv("rf_results.csv")
print("##################")