import argparse
import cProfile as profile
import glob
import os

import cv2
import numpy as np
import scipy.io as sio
import pandas as pd

from misc.info import MAP_TYPES

from metrics.stats_utils import *


def run_nuclei_type_stat(
    pred_dir, true_dir, nuclei_type_dict, type_uid_list=None, exhaustive=True, rad=12, verbose=False
):
    """
    rad = 12 if x40
    rad = 6 if x20
    """
    def _get_type_name(uid, ntd=nuclei_type_dict):
        for name,v in ntd.items():
            if v == uid:
                return name

    def calc_type_metrics(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        # unpaired_pred_t = unpaired_pred[unpaired_pred == type_id] # (unpaired_pred == type_id).sum()
        # unpaired_true_t = unpaired_true[unpaired_true == type_id]

        # Original
        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        # Classification
        # TP - detected cell with GT label t, classified as t
        tp_dtc = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        # TN - detected cell with GT label other than t, classified as other than t
        tn_dtc = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        # FP - detected cell with GT label other than t classified as t
        fp_dtc = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        # FN - detected cell with GT label t classified as other than t
        fn_dtc = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        # Integrated classification
        # TP - detected cell with GT label t, classified as t
        tp_dtic = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        
        # TN - detected or falsely detected cell with GT label other than t, classified as other than t
        tn_dtic = np.concatenate((
            ((paired_true != type_id) & (paired_pred != type_id)),
            (unpaired_pred != type_id)
            # np.concatenate(
            #     ((unpaired_true != type_id), (unpaired_pred != type_id))
            # )
        )).sum()

        # FP - detected or falsely detected cell with GT label other than t, classified as t
        fp_dtic = np.concatenate((
            ((paired_true != type_id) & (paired_pred == type_id)),
            (unpaired_pred == type_id)
            # np.concatenate(
            #     ((unpaired_true != type_id), (unpaired_pred == type_id))
            # )
        )).sum()

        # FN - detected cell with GT label t, classified as other than t and all cells with GT label t not detected
        fn_dtic = np.concatenate((
            ((paired_true == type_id) & (paired_pred != type_id)),
            (unpaired_true == type_id)
        )).sum()

        if not exhaustive:
            ignore = (paired_true == -1).sum()
            fp_dt -= ignore

        tp_d = (paired_pred == type_id).sum()
        # tn_d = (paired_true == type_id).sum()
        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()
        
        rec_dt = tp_d / (tp_d + fn_d)

        def __internal_metrics(tp, tn, fp, fn):
            # print (f"tp: {tp}, \ntn: {tn}, \nfp:{fp}, fn: {fn}\n")
            acc = (tp + tn) / (tp + fp + fn + tn)
            prec = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (prec * recall) / (prec + recall)
            # print (f"Accuracy: {acc}, \nPrecision: {prec}, \nRecall:{recall}, F1: {f1}\n")
            return acc, prec, recall, f1

        res_class = __internal_metrics(tp_dtc, tn_dtc, fp_dtc, fn_dtc)
        dtc_tptnfpfn = (tp_dtc, tn_dtc, fp_dtc, fn_dtc)
        res_i_class = __internal_metrics(tp_dtic, tn_dtic, fp_dtic, fn_dtic)
        dtic_tptnfpfn = (tp_dtic, tn_dtic, fp_dtic, fn_dtic)

        # print (f"tp_dt: {tp_dt}") # TPc
        # print (f"tn_dt: {tn_dt}") # TNc
        # print (f"fp_dt: {fp_dt}") # FPc
        # print (f"fn_dt: {fn_dt}") # FNc
        # print (f"fp_d: {fp_d}")
        # print (f"fn_d: {fn_d}")

        tp_w = tp_dt + tn_dt
        fp_w = 2 * fp_dt + fp_d
        fn_w = 2 * fn_dt + fn_d

        w_f1_type = (2 * (tp_dt + tn_dt)) / (
            2 * (tp_dt + tn_dt)
            + w[0] * fp_dt
            + w[1] * fn_dt
            + w[2] * fp_d
            + w[3] * fn_d
        )
        w_acc_type = (tp_w) / (tp_w + fp_w + fn_w) ## check

        w_precision_type = tp_w / (tp_w + fp_w)
        w_recall_type = tp_w / (tp_w + fn_w)

        weighted = (w_acc_type, w_precision_type, w_recall_type, w_f1_type)

        cls_r = (dtc_tptnfpfn, res_class)
        icls_r = (dtic_tptnfpfn, res_i_class)

        #return f1_type, precision_type, recall_type
        return (
            rec_dt, ### Segmentation recall
            cls_r, ### Classification
            icls_r, ### Integrated classification
            weighted ### Weighted
        )

    ######################################################
    types = sorted([f"{v}:{k}" for k, v in nuclei_type_dict.items()])
    if verbose: print(types)

    file_list = glob.glob(os.path.join(pred_dir, "*.mat"))
    file_list.sort()  # ensure same order [1]
    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point

    for file_idx, filename in enumerate(file_list[:]):
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]
        # print (basename)
        # true_info = sio.loadmat(os.path.join(true_dir, '{}.mat'.format(basename)))
        # # dont squeeze, may be 1 instance exist
        # true_centroid  = (true_info['inst_centroid']).astype('float32')
        # true_inst_type = (true_info['inst_type']).astype('int32')

        true_info = np.load(
            os.path.join(true_dir, "{}.npy".format(basename)), allow_pickle=True
        )
        # dont squeeze, may be 1 instance exist
        true_centroid = (true_info.item().get("inst_centroid")).astype("float32")
        true_inst_type = (true_info.item().get("inst_type")).astype("int32")
        if true_centroid.shape[0] != 0:
            true_inst_type = true_inst_type[:, 0]
        else:  # no instance at all
            pass
            true_centroid = np.array([[0, 0]])
            true_inst_type = np.array([0])

        pred_info = sio.loadmat(os.path.join(pred_dir, "{}.mat".format(basename)))
        # dont squeeze, may be 1 instance exist
        pred_centroid = (pred_info["inst_centroid"]).astype("float32")
        pred_inst_type = (pred_info["inst_type"]).astype("int32")

        if pred_centroid.shape[0] != 0:
            pred_inst_type = pred_inst_type[:, 0]
        else:  # no instance at all
            pass
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroid, pred_centroid, rad
        )
        
        # * Aggreate information
        # get the offset as each index represent 1 independent instance
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if file_idx != 0 else 0
        )
        true_inst_type_all.append(true_inst_type)
        pred_inst_type_all.append(pred_inst_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)

    paired_all = np.concatenate(paired_all, axis=0) # (x, 2) # paired ids (found in GT and pred)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0) # (x,) # unpaired ids (found in GT and NOT in pred)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0) # (x,) # unpaired ids (NOT found in GT and found in pred)

    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0) # all type ids in true [3,3,3...1,1,1]
    paired_true_type = true_inst_type_all[paired_all[:, 0]] # paired true type ids [3,3,3...1,1,1]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]

    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0) # all type ids in pred [3,3,3...1,1,1]
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]
    # true_inst_type_all = paired_true_type + unpaired_true_type

    ###
    # overall
    # * quite meaningless for not exhaustive annotated dataset
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0]
    fn_d = unpaired_true_type.shape[0]

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    if not exhaustive:
        ignore = (paired_true_type == -1).sum()
        fp_fn_dt -= ignore
    
    w = [1, 1]
    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
    precision = tp_d / (tp_d + w[0] * fp_d)
    recall = tp_d / (tp_d + w[0] * fn_d)
    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

    # results_list = [acc_type, precision, recall, f1_d]

    results_all_types = [[acc_type], [precision], [recall], [f1_d]]

    w = [2, 2, 1, 1]

    if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()
        if 0 in type_uid_list:
            type_uid_list.remove(0)
    
    pred_type_uid_list = np.unique(pred_inst_type_all).tolist()
    if 0 in pred_type_uid_list:
        pred_type_uid_list.remove(0)

    if verbose:
        print(f"True type_uid_list: {type_uid_list}")
        print(f"Pred type_uid_list: {pred_type_uid_list}")

    res_all = {}
    for type_uid in type_uid_list:
        res = calc_type_metrics(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w,
        )
        result_uid_metrics = [
            [res[0]], # rec_dt ### Segmentation recall
            [res[1][1][0]], [res[1][1][1]], [res[1][1][2]], [res[1][1][3]], # (dtc_tptnfpfn, res_class), ### Classification
            [res[2][1][0]], [res[2][1][1]], [res[2][1][2]], [res[2][1][3]], # (dtic_tptnfpfn, res_i_class), ### Integrated classification
            [res[3][0]], [res[3][1]], [res[3][2]], [res[3][3]] # weighted ### Weighted
        ]
        res_all[f"{type_uid}:{_get_type_name(type_uid)}"] = result_uid_metrics

    ### I - integrated, W - weighted, Type - across all types
    cols_uid = ["Recall_dt", "Cls_acc", "Cls_precision", "Cls_recall", "Cls_F1", "ICls_acc", "ICls_precision", "ICls_recall", "ICls_F1", "WCls_acc", "WCls_precision", "WCls_recall", "WCls_F1"] # result_uid_metrics
    cols_all_types = ["Type_acc", "Type_precision", "Type_recall", "Type_F1"] # results_all_types

    df_all_types = pd.DataFrame(np.transpose(np.array(results_all_types)), columns=cols_all_types)
    
    df_uid = pd.DataFrame(np.squeeze(np.array(list(res_all.values()))), columns=cols_uid)
    df_uid["Type"] = list(res_all.keys())
    df_uid = df_uid[["Type", *cols_uid]]

    if verbose:
        print()
        print(df_all_types.to_markdown(index=False))
        print()
        print(df_uid.to_markdown(index=False))

    return df_uid, df_all_types


def run_nuclei_inst_stat(pred_dir, true_dir, print_img_stats=False):
    
    file_list = glob.glob(os.path.join(pred_dir, "*.mat"))
    file_list.sort()  # ensure same order

    metrics = [[], [], [], [], [], []]
    cols = ["dice", "fast_aji", "dq", "sq", "pq", "aji_plus"]
    names = []

    for filename in file_list[:]:
        filename = os.path.basename(filename)
        basename = filename.split(".")[0]
        names.append(basename)

        # true = sio.loadmat(os.path.join(true_dir, '{}.mat'.format(basename)))
        # true = (true['inst_map']).astype('int32')
        true = np.load(
            os.path.join(true_dir, "{}.npy".format(basename)), allow_pickle=True
        )
        true = (true.item().get("inst_map")).astype("int32")

        pred = sio.loadmat(os.path.join(pred_dir, "{}.mat".format(basename)))
        pred = (pred["inst_map"]).astype("int32")

        # to ensure that the instance numbering is contiguous
        pred = remap_label(pred, by_size=False)
        true = remap_label(true, by_size=False)

        pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
        metrics[0].append(get_dice_1(true, pred))
        metrics[1].append(get_fast_aji(true, pred))
        metrics[2].append(pq_info[0])  # dq
        metrics[3].append(pq_info[1])  # sq
        metrics[4].append(pq_info[2])  # pq
        metrics[5].append(get_fast_aji_plus(true, pred))

    metrics = np.array(metrics)
    metrics_per_patient = pd.DataFrame(np.transpose(metrics), columns=cols)
    metrics_per_patient["filename"] = names
    metrics_per_patient = metrics_per_patient[["filename", *cols]]
    if print_img_stats: print(metrics_per_patient.to_markdown(index=False))

    metrics_avg = np.mean(metrics, axis=-1)
    metrics_std = np.std(metrics, axis=-1)

    metrics_avg = pd.DataFrame(np.array([metrics_avg, metrics_std]), columns=cols)
    metrics_avg["type"] = ["avg", "std"]
    metrics_avg = metrics_avg[["type", *cols]]
    if print_img_stats: print(metrics_avg.to_markdown(index=False))

    return metrics_per_patient, metrics_avg



if __name__ == "__main__":
    # cfg = Config(verbose=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", help="Point to output dir", required=True)
    parser.add_argument("--true_dir", help="Point to ground truth dir", required=True)
    parser.add_argument(
        "--map",
        help="Name of the nuclei type mapping (mapping used by model inferece)",
        choices=["consep", "pannuke", "monusac"],
        required=True,
    )
    parser.add_argument(
        "--type", help="Run type stats", default=False, action="store_true"
    )
    parser.add_argument(
        "--inst", help="Run inst stats", default=False, action="store_true"
    )
    args = parser.parse_args()

    if args.type:
        print("---Type statistics---")
        df_uid, df_all_types = run_nuclei_type_stat(
            args.pred_dir, args.true_dir, MAP_TYPES[f"hv_{args.map}"], rad=12, verbose=True
        )
    if args.inst:
        print("---Instance statistics---")
        df_per_patient, df_avg = run_nuclei_inst_stat(args.pred_dir, args.true_dir, print_img_stats=True)

# Combined processing logic from process.py
import glob
import os
import argparse
from datetime import datetime

import cv2
import numpy as np
import scipy.io as sio
from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)
from skimage.morphology import remove_small_objects, watershed

import postproc.hover

from config import Config

from misc.viz_utils import visualize_instances
from misc.utils import get_inst_centroid
from metrics.stats_utils import remap_label

from joblib import Parallel, delayed

AV_CPU = os.cpu_count()

###################

def swap_classes(pred, mapping):
    '''
    # Example: change 1 to 4 and 2 to 3.
    mapping = {1:4, 2:3}
    '''
    id_map = pred.copy()
    for k,v in mapping.items():
        pred[id_map == k] = v
    return pred


def process(jobs):

    ## ! WARNING:
    ## check the prediction channels, wrong ordering will break the code !
    ## the prediction channels ordering should match the ones produced in augs.py

    cfg = Config()

    # * flag for HoVer-Net only
    # 1 - threshold, 2 - sobel based
    energy_mode = 2
    marker_mode = 2

    for num, data_dir in enumerate(cfg.inf_data_list):
        pred_dir = os.path.join(cfg.inf_output_dir, str(num))
        proc_dir = "{}_processed".format(pred_dir)

        file_list = glob.glob(os.path.join(pred_dir, "*.mat"))
        file_list.sort()  # ensure same order

        if not os.path.isdir(proc_dir):
            os.makedirs(proc_dir)

        def process_image(filename):
            filename = os.path.basename(filename)
            basename = filename.split(".")[0]
            if jobs == 1:
                print(pred_dir, basename, flush=True)

            ##
            img = cv2.imread(
                os.path.join(data_dir, "{}{}".format(basename, cfg.inf_imgs_ext))
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pred = sio.loadmat(os.path.join(pred_dir, "{}.mat".format(basename)))
            pred = np.squeeze(pred["result"])

            pred_inst = pred[..., cfg.nr_types :]
            pred_type = pred[..., : cfg.nr_types]

            pred_inst = np.squeeze(pred_inst)
            pred_type = np.argmax(pred_type, axis=-1)

            if cfg.model_type == "np_hv" or cfg.model_type == "np_hv_opt":
                pred_inst = postproc.hover.proc_np_hv(
                    pred_inst, marker_mode=marker_mode, energy_mode=energy_mode, rgb=img
                )

            # ! will be extremely slow on WSI/TMA so it's advisable to comment this out
            # * remap once so that further processing faster (metrics calculation, etc.)
            if cfg.remap_labels:
                pred_inst = remap_label(pred_inst, by_size=True)

            #### * Get class of each instance id, stored at index id-1
            pred_id_list = list(np.unique(pred_inst))[1:]  # exclude background ID
            pred_inst_type = np.full(len(pred_id_list), 0, dtype=np.int32)
            for idx, inst_id in enumerate(pred_id_list):
                inst_type = pred_type[pred_inst == inst_id]
                type_list, type_pixels = np.unique(inst_type, return_counts=True)
                type_list = list(zip(type_list, type_pixels))
                type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
                inst_type = type_list[0][0]
                if inst_type == 0:  # ! pick the 2nd most dominant if exist
                    if len(type_list) > 1:
                        inst_type = type_list[1][0]
                    else:
                        if jobs == 1:
                            pass  # print('[Warn] Instance has `background` type')
                pred_inst_type[idx] = inst_type
            pred_inst_centroid = get_inst_centroid(pred_inst)


            if cfg.process_mapping is not None:
                pred_type = swap_classes(pred_type, cfg.process_mapping)
                pred_inst_type = swap_classes(pred_inst_type, cfg.process_mapping)

            ###### ad hoc for pannuke predictions
            # 5 -> 4

            ###### ad hoc for squash_monusac predictions
            # 3 -> 2
            # 4 -> 2

            ###### ad hoc for consep model to monusac data
            # 3 -> 2
            # 4 -> 2
            # 1 -> 3

            ###### ad hoc for monusac to consep predictions
            # 3 -> 2
            # 4 -> 2
            # 1 -> 3

            ###### ad hoc for monusac to pannuke predictions
            # 1 -> 4
            # 2 -> 1
            # 3 -> 1
            # 4 -> 1
            # 5 -> 4


            # print (np.unique(pred_type))
            # if 0 not in np.unique(pred_type):
            #     return

            sio.savemat(
                os.path.join(proc_dir, "{}.mat".format(basename)),
                {
                    "inst_map": pred_inst,
                    "type_map": pred_type,
                    "inst_type": pred_inst_type[:, None],
                    "inst_centroid": pred_inst_centroid,
                },
            )
            overlaid_output = visualize_instances(
                pred_inst,
                img,
                ((cfg.nuclei_type_dict, cfg.color_palete), pred_inst_type[:, None]),
                cfg.outline,
                cfg.skip_types,
            )
            overlaid_output = cv2.cvtColor(overlaid_output, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                os.path.join(proc_dir, "{}.png".format(basename)), overlaid_output
            )
            with open(os.path.join(proc_dir, f"{basename}.log"), "w") as log_file:
                unique, counts = np.unique(
                    pred_inst_type[:, None], return_counts=True
                )
                unique = list(unique)
                if 0 in unique:  # remove backround entries
                    counts = np.delete(counts, unique.index(0))
                    unique.remove(0)
                print(
                    f"{basename} : {dict(zip([{str(v): str(k) for k, v in cfg.nuclei_type_dict.items()}[str(item)] for item in unique], counts))}",
                    file=log_file,
                )

            ##
            if jobs == 1:
                print(
                    f"Finished for {basename} {datetime.now().strftime('%H:%M:%S.%f')}"
                )

        start = datetime.now()
        if jobs > 1:
            assert (
                AV_CPU > jobs
            ), f"Consider using lower number of jobs (recommended {AV_CPU - 4 if AV_CPU - 4 > 0 else 1})"
            print(
                f"Stared parallel process for ```{data_dir}``` {start.strftime('%d/%m %H:%M:%S.%f')}"
            )
            print(f"Using {jobs} jobs")
            Parallel(n_jobs=jobs, prefer="threads")(
                delayed(process_image)(filename) for filename in file_list
            )
            end = datetime.now()
            print(
                f"Done parallel process for ```{data_dir}``` {end.strftime('%d/%m %H:%M:%S.%f')}"
            )
        else:
            for filename in file_list:
                process_image(filename)
            end = datetime.now()
        print(f"Overall time elapsed: {end - start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", help="How many jobs used to run processing", type=int, default=1
    )
    args = parser.parse_args()
    process(args.jobs)
