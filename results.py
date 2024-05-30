import json
import numpy as np
import os
from numpy import array as npa
import cv2 
from utils_various import *
from PIL import Image
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors  # used for coloring bins in histograms
import matplotlib.patches as patches
from matplotlib.ticker import PercentFormatter  # used for setting y-axis tick
                                                # to percent of test images
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from cycler import cycler  # used for changing default color sequence @ pyplot
plt.rc('text.latex', preamble=r'\usepackage{amsmath}'
                              r'\usepackage{amsfonts}')
mpl.rcParams['font.size'] = 12
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'large'
mpl.rcParams['xtick.labelsize'] = 'medium'
mpl.rcParams['legend.framealpha'] = 0.65
mpl.rcParams['axes.labelsize'] = 'large'

def get_pose_error(r_GT, q_GT, r_pred, q_pred):
    Et_vec = abs(r_GT - r_pred)
    Et = np.linalg.norm(r_GT - r_pred, 2, axis=1)
    et = Et/np.linalg.norm(r_GT, 2, axis=1)
    eq = 2*np.arccos(np.minimum(1, np.abs(np.sum(q_GT*q_pred, axis=1)/(np.linalg.norm(q_GT, 2, axis=1)*np.linalg.norm(q_pred, 2, axis=1)))))
    return Et_vec, Et, et, eq

def plot_wireframe(dcm, r, keypoints, bbox, ax, imgname, err_r, euler_angles, err_euler, line_color='cyan', line_width=1):
    if ax is None:
        ax = plt.gca()
    # get landmarks corresponding to projection of predicted pose
    xl, yl = project3Dto2D(dcm, r, Wireframe.landmark_mat)
    img = cv2.imread("test/images/" + imgname)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor=(185/255,5/255,4/255), facecolor="none")

    ax.add_patch(rect)
    # bottom of main body
    plt.plot(np.hstack((xl[0:4], xl[0])),
             np.hstack((yl[0:4], yl[0])),
             color=(108/255, 196/255, 193/255), lw=line_width)
    plt.xlim([0, Camera.nu])
    plt.ylim([Camera.nv, 0])
    plt.xticks([])
    plt.yticks([])
    # solar panel
    plt.plot(np.hstack((xl[4:8], xl[4])),
             np.hstack((yl[4:8], yl[4])),
             color=(108/255, 196/255, 193/255), lw=line_width)
    # top of main body
    x_top, y_top = project3Dto2D(dcm, r, Wireframe.topMainBody_mat)
    plt.plot(np.hstack((x_top[0:4], x_top[0])),
             np.hstack((y_top[0:4], y_top[0])),
             color=(108/255, 196/255, 193/255), lw=line_width)
    # corners
    for k in np.arange(0, 4):
        plt.plot(np.hstack((xl[k], x_top[k])),
                 np.hstack((yl[k], y_top[k])),
                 color=(108/255, 196/255, 193/255), lw=line_width)
    # antennas
    x_clamp, y_clamp = project3Dto2D(dcm, r, Wireframe.antClamps_mat)
    for k in np.arange(0, 3):
        plt.plot(np.hstack((x_clamp[k], xl[8 + k])),
                 np.hstack((y_clamp[k], yl[8 + k])),
                 color=(108/255, 196/255, 193/255), lw=line_width)
    for i in range(11):
        cv2.circle(img, (int(keypoints[i, 0]), int(keypoints[i, 1])), radius = 10, color = (108, 196, 193), thickness=-1)
    ax.imshow(img)
    
    box_propeties = dict(boxstyle='round', facecolor='gray', alpha=0.8, edgecolor='gray')
    ax.text(0.03, 0.96,
            '$\\bf{Distance}$ [m]\n\t$x$ = %.2f\n\t$y$ = %.2f\n\t$z$ = %.2f'
            % (r[0], r[1], r[2])
            + '\n\nError [m]:\n\t$E_x = %.3f$\n\t$E_y = %.3f$\n\t$E_z = %.3f$'
            % (err_r[0], err_r[1], err_r[2]),
            transform=ax.transAxes, fontsize=10, color="w", alpha=1,
                verticalalignment='top', bbox=box_propeties)
    ax.text(0.03, 0.5,
            '$\\bf{Attitude}$\n\t$\\theta_x = %.2f^\circ$\n\t$\\theta_y = %.2f^\circ$\n\t$\\theta_z = %.2f^\circ$'
            % (euler_angles[0], euler_angles[1], euler_angles[2])
            + '\n\nError: \n\t$E_{\\theta_x} = %.3f^\circ$\n\t$E_{\\theta_y} = %.3f^\circ$'
                '\n\t$E_{\\theta_z} = %.3f^\circ$'
            % (err_euler[0], err_euler[1], err_euler[2]),
            transform=ax.transAxes, fontsize=10, color="w", alpha=1,
            verticalalignment='top', bbox=box_propeties)


    # REMOVES ANY WHITESPACES FROM THE SAVED FIGURE
    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())



with open("./output/param_optimization.json", "r") as f:
    param_opt = f.read()

js = json.loads(param_opt)

num_LM = np.array(js["MIN_LANDMARK"])
conf_LM = np.array(js["MIN_CONFIDENCE"])
score = np.array(js["SLAB_score"])

fig = plt.figure()
ax = plt.gca()
im = ax.imshow(score, extent = [conf_LM[0] - 0.025, conf_LM[-1] + 0.025, num_LM[0] - 0.5, num_LM[-1] + 0.5],
               origin = "lower", interpolation = "antialiased", aspect=0.1, vmin=np.min(score), vmax=np.max(score))
cbar = fig.colorbar(im)
cbar.ax.set_title('$\\rm{e_{SLAB}}$')  # Median Normalized Pose Error

plt.xlabel('Landmark threshold confidence')
plt.ylabel('Minimum # LM')

# Best (minimum) score

ij_min = np.where(score == np.min(score))
conf_opt = conf_LM[ij_min[1][0]]
numLM_opt = num_LM[ij_min[0][0]]
print("Optimum conf score: ", conf_opt)


plt.plot(conf_opt, numLM_opt,
         marker='D', markeredgecolor='black', fillstyle='none', markeredgewidth=1.5)
box_propeties = dict(boxstyle='round', facecolor='none', alpha=0.8,
                     edgecolor='b', linestyle='-', linewidth=1.5)
fig.savefig(os.path.join('output', 'landmark_rejection_optimization.pdf'), bbox_inches='tight')
print("Optimum min LM: ", numLM_opt)


result_file = "output/inf_results.json"
with open(result_file, "r") as f:
    data = f.read()
js_pred = json.loads(data)


with open("test/test.json") as f:
    data = f.read()
js_GT = json.loads(data)

with open("output/time_results_16.json") as f:
    data = f.read()
time = json.loads(data)

r_pred = np.zeros((len(js_pred), 3))
q_pred = np.zeros((len(js_pred), 4))
t_det = np.zeros((len(js_pred), 1))
t_key = np.zeros((len(js_pred), 1))
t_epnp = np.zeros((len(js_pred), 1))
r_GT = np.zeros((len(js_pred), 3))
q_GT = np.zeros((len(js_pred), 4))
keypoints = np.zeros((11, 2, len(js_pred)))
bbox = np.zeros((len(js_pred), 4))
imgname = []
outlier = []
for i, _ in enumerate(js_pred):
    imgname.append(js_GT[i]["filename"])
    r_pred[i] = npa(js_pred[i]["tvec"])
    q_pred[i] = npa(js_pred[i]["quat"])
    t_det[i] = npa(time[i]["det_time"])
    t_key[i] = npa(time[i]["key_time"])
    t_epnp[i] = npa(js_pred[i]["epnp_time"])
    r_GT[i] = npa(js_GT[i]["r_Vo2To_vbs_true"])
    q_GT[i] = npa(js_GT[i]["q_vbs2tango"])
    #outlier.append((js_pred[i]["outlier"]))
    keypoints[:,:,i] = npa(js_pred[i]["keypoints"])
    bbox[i] = npa(js_pred[i]["bbox"])


Et_vec, Et, et, eq = get_pose_error(r_GT, q_GT, r_pred, q_pred)

print("Mean translation error (x, y, z) [m]: ", np.mean(Et_vec, axis=0))
print("Mean translation error (norm) [m]: ", np.mean(Et))
print("Mean normalized translation error: ", np.mean(et))
print("Mean rotation error [deg]: ", np.rad2deg(np.mean(eq)))
print("SLAB score: ", np.mean(et + eq))

print("\nMean det time [s]: ", np.mean(t_det))
print("Mean land reg time [s]: ", np.mean(t_key))
print("Mean EPnP time: ", np.mean(t_epnp))
print("Mean total time: ", np.mean(t_det + t_key + t_epnp))

# Worst and best score:
minID = np.argmin(eq + et)
maxID = np.argmax(eq + et)
print("\nBest image: ", imgname[minID], ". Score: ", (et+eq)[minID])
print("Worst image: ", imgname[maxID], ". Score: ", (et+eq)[maxID])
# Outlier detected:
#print("\nOutlier detected in images: ", [name for i,name in enumerate(imgname) if outlier[i]])
fig = plt.figure()
ax = plt.gca()
plot_wireframe(quat2dcm(q_pred[maxID]), r_pred[maxID], keypoints[:,:,maxID], bbox[maxID], ax, 
               imgname[maxID], Et_vec[maxID], dcm2euler(quat2dcm(q_pred[maxID])), abs(dcm2euler(quat2dcm(q_pred[maxID])) - dcm2euler(quat2dcm(q_GT[maxID]))))

fig = plt.figure()
ax = plt.gca()
plot_wireframe(quat2dcm(q_pred[minID]), r_pred[minID], keypoints[:,:,minID], bbox[minID], ax, imgname[minID],
               Et_vec[minID], dcm2euler(quat2dcm(q_pred[minID])), abs(dcm2euler(quat2dcm(q_pred[minID])) - dcm2euler(quat2dcm(q_GT[minID]))))

# Inference time plot
fig = plt.figure(figsize=(13,6))
ax = fig.gca()
ax.plot(range(len(t_det)), np.concatenate((t_det, t_key, t_epnp, t_det+t_key+t_epnp), axis=1))
ax.legend(labels = ("$t_{det}$", "$t_{key}$", "$t_{EPnP}$", "$t_{tot}$"))
ax.set_xticks(np.linspace(0, 2400, 6))
ax.set_yscale("log")
plt.xlabel("Image #")
plt.ylabel("Runtime [s]")
plt.grid(which="both")
ID_sort = np.argsort(eq + et)

# Error vs GT-distance
idx_sortDist = np.argsort(np.linalg.norm(r_GT, 2, 1))
r_GT_norm_sort = np.linalg.norm(r_GT, 2, 1)[idx_sortDist]
Et_sort = Et[idx_sortDist]
eq_sort = np.rad2deg(eq[idx_sortDist])
dist_step = 80  # i.e. we compute the mean over batches of 100 images
dist_axis = np.hstack(
    (r_GT_norm_sort[np.arange(0, len(r_GT_norm_sort), dist_step)], r_GT_norm_sort[-1])
)
perc_bounds = [15.87, 84.13]  # i.e. 1-sigma range

def compute_error_percentiles_vs_distance(dist_axis, dist_gt_sorted, err_vec, perc_bounds):
    perc1_vec = []
    perc2_vec = []
    mean_err = []
    mean_dist = []
    for k in np.arange(len(dist_axis)-1):
        d1 = dist_axis[k]
        d2 = dist_axis[k+1]
        idx_distStep = np.logical_and(dist_gt_sorted >= d1, dist_gt_sorted <= d2)

        mean_dist.append( np.mean(dist_gt_sorted[idx_distStep]) )

        values_in_dist_range = err_vec[idx_distStep]
        mean_err.append( np.mean(values_in_dist_range) )

        perc1_vec.append( np.percentile(values_in_dist_range, perc_bounds[0]) )
        perc2_vec.append( np.percentile(values_in_dist_range, perc_bounds[1]) )

    return npa(mean_dist), npa(mean_err), npa(perc1_vec), npa(perc2_vec)

# ABSOLUTE ERRORS
# Translation error
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mean_dist, mean_err, p1, p2, = compute_error_percentiles_vs_distance(
    dist_axis=dist_axis, dist_gt_sorted=r_GT_norm_sort,
    err_vec=Et_sort*100, perc_bounds=perc_bounds
)

fig, ax = plt.subplots(2, figsize=(13,6))
ax[0].fill_between(mean_dist, p1, p2, alpha=0.3, ec='none', fc=colors[0])
ax[0].plot(mean_dist,mean_err, '-o', color=colors[0], fillstyle='none', markersize=5)
ax[0].legend(['Mean error','1$\sigma$ range'][::-1], loc=2)
ax[0].set_ylabel('Translation error [cm]')
ax[0].grid(True)
ax[0].set_axisbelow(True)

# Rotation error
mean_dist, mean_err, p1, p2, = compute_error_percentiles_vs_distance(
    dist_axis=dist_axis, dist_gt_sorted=r_GT_norm_sort,
    err_vec=eq_sort, perc_bounds=perc_bounds
)

ax[1].fill_between(mean_dist, p1, p2, alpha=0.3, ec='none', fc=colors[1])
ax[1].plot(mean_dist, mean_err, '-o', color=colors[1], fillstyle='none', markersize=5)
ax[1].legend(['Mean error','1$\sigma$ range'][::-1], loc=2)
ax[1].set_xlabel('Mean true distance [m]')
ax[1].set_ylabel('Quaternion error [deg]')
ax[1].grid(True)
ax[1].set_axisbelow(True)

# POSE SCORES
fig = plt.figure(figsize=[13,6])
ax = fig.gca()

# SLAB score
mean_dist, mean_err, p1, p2, = compute_error_percentiles_vs_distance(
    dist_axis=dist_axis, dist_gt_sorted=r_GT_norm_sort,
    err_vec=(et[idx_sortDist] + eq[idx_sortDist]), perc_bounds=perc_bounds
)

ax.fill_between(mean_dist, p1, p2, alpha=0.3, ec='none', fc=colors[2])
ax.plot(mean_dist, mean_err, '-o', color=colors[2], fillstyle='none', markersize=5)
ax.set_ylabel('SLAB score [-]')
ax.set_xlabel('Mean true distance [m]')
ax.grid(True)
ax.set_axisbelow(True)
_, mean_err, _ , _ = compute_error_percentiles_vs_distance(
    dist_axis=dist_axis, dist_gt_sorted=r_GT_norm_sort,
    err_vec=(et[idx_sortDist]), perc_bounds=perc_bounds
)
ax.plot(mean_dist, mean_err, '-s', color=colors[0], fillstyle='none', markersize=5)
_, mean_err, _ , _ = compute_error_percentiles_vs_distance(
    dist_axis=dist_axis, dist_gt_sorted=r_GT_norm_sort,
    err_vec=(eq[idx_sortDist]), perc_bounds=perc_bounds
)
ax.plot(mean_dist, mean_err, '-d', color=colors[1], fillstyle='none', markersize=5)
ax.legend(['1$\sigma$ range (total error)', 'Mean total error', 'Translation error ($e_t$)',
              'Rotation error ($E_q$)'], loc=2)



# ERROR VS EARTH BACKGROUND

no_EARTH_error = []
EARTH_error = []

import re
for i in range(len(imgname)):
    img_idx = int(re.search(r'\d+', imgname[i]).group())
    if img_idx > earthBGStart_ID:
        EARTH_error.append((et+eq)[i])
    else:
        no_EARTH_error.append((et+eq)[i])

print("Mean Earth error: ", np.mean(EARTH_error), " STD: ", np.std(EARTH_error))
print("Mean no Earth error: ", np.mean(no_EARTH_error), " STD: ", np.std(no_EARTH_error))

labels = ['Earth Background', 'Black Background']
means = [np.mean(EARTH_error), np.mean(no_EARTH_error)]  # Replace with your mean values
std_devs = [np.std(EARTH_error), np.std(no_EARTH_error)]   # Replace with your standard deviation values



fig = plt.figure()
ax = plt.gca()
for i in range(len(imgname)):
    ax.clear()
    plot_wireframe(quat2dcm(q_pred[i]), r_pred[i], keypoints[:,:,i], bbox[i], ax, imgname[i],
               Et_vec[i], dcm2euler(quat2dcm(q_pred[i])), abs(dcm2euler(quat2dcm(q_pred[i])) - dcm2euler(quat2dcm(q_GT[i]))))
    fig.savefig("./output/visualization/" + imgname[i].replace("jpg", "pdf"), bbox_inches='tight', pad_inches=0)