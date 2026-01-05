# Evaluates semantic instance task
# Adapted from the CityScapes evaluation: https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/evaluation
# Input:
#   - path to .txt prediction files
#   - path to .txt ground truth files
#   - output file to write results to
# Each .txt prediction file look like:
#    [(pred0) rel. path to pred. mask over verts as .txt] [(pred0) label id] [(pred0) confidence]
#    [(pred1) rel. path to pred. mask over verts as .txt] [(pred1) label id] [(pred1) confidence]
#    [(pred2) rel. path to pred. mask over verts as .txt] [(pred2) label id] [(pred2) confidence]
#    ...
#
# NOTE: The prediction files must live in the root of the given prediction path.
#       Predicted mask .txt files must live in a subfolder.
#       Additionally, filenames must not contain spaces.
# The relative paths to predicted masks must contain one integer per line,
# where each line corresponds to vertices in the *_vh_clean_2.ply (in that order).
# Non-zero integers indicate part of the predicted instance.
# The label ids specify the class of the corresponding mask.
# Confidence is a float confidence score of the mask.
#
# Note that only the valid classes are used for evaluation,
# i.e., any ground truth label not in the valid label set
# is ignored in the evaluation.
#
# example usage: evaluate_semantic_instance.py --scan_path [path to scan data] --output_file [output file]

# python imports
import math
import logging
from pathlib import Path
import os, sys, argparse
import inspect
from copy import deepcopy
from uuid import uuid4

import torch

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

from scipy import stats

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
import benchmark.util as util
import benchmark.util_3d as util_3d

# parser = argparse.ArgumentParser()
# parser.add_argument('--gt_path', default='', help='path to directory of gt .txt files')
# parser.add_argument('--output_file', default='', help='output file [default: ./semantic_instance_evaluation.txt]')
# opt = parser.parse_args()

# if opt.output_file == '':
#    opt.output_file = os.path.join(os.getcwd(), 'semantic_instance_evaluation.txt')


# ---------- Label info ---------- #
CLASS_LABELS = [
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherstructure",
    "otherfurniture",
    # add 2 more classes for scannet in case we want to train on them
    "otherprop",
]
# add 2 more classes for scannet in case we want to train and evalute on them
# add otherstructure and otherprop in the correct places so that the final IDs are correct?
VALID_CLASS_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 38, 39, 40]
)
ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
# ---------- Evaluation params ---------- #
# overlaps for evaluation
opt = {}
opt["overlaps"] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
# minimum region size for evaluation [verts]
opt["min_region_sizes"] = np.array([100])  # 100 for s3dis, scannet
# distance thresholds [m]
opt["distance_threshes"] = np.array([float("inf")])
# distance confidences
opt["distance_confs"] = np.array([-float("inf")])


def evaluate_matches(matches):
    overlaps = opt["overlaps"]
    min_region_sizes = [opt["min_region_sizes"][0]]
    dist_threshes = [opt["distance_threshes"][0]]
    dist_confs = [opt["distance_confs"][0]]

    # results: class x overlap
    ap = np.zeros(
        (len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float
    )
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
        zip(min_region_sizes, dist_threshes, dist_confs)
    ):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for p in matches[m]["pred"]:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]["pred"][label_name]:
                            if "uuid" in p:
                                pred_visited[p["uuid"]] = False
            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                for m in matches:
                    pred_instances = matches[m]["pred"][label_name]
                    gt_instances = matches[m]["gt"][label_name]
                    # filter groups in ground truth
                    gt_instances = [
                        gt
                        for gt in gt_instances
                        if gt["instance_id"] >= 1000
                        and gt["vert_count"] >= min_region_size
                        and gt["med_dist"] <= distance_thresh
                        and gt["dist_conf"] >= distance_conf
                    ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)
                    # collect matches
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        num_pred = len(gt["matched_pred"])
                        for pred in gt["matched_pred"]:
                            # greedy assignments
                            if pred_visited[pred["uuid"]]:
                                continue
                            overlap = float(pred["intersection"]) / (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - pred["intersection"]
                            )
                            if overlap > overlap_th:
                                confidence = pred["confidence"]
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred["uuid"]] = True
                        if not found_match:
                            hard_false_negatives += 1
                    # remove non-matched ground truth instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # collect non-matched predictions as false positive
                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred["matched_gt"]:
                            overlap = float(gt["intersection"]) / (
                                gt["vert_count"]
                                + pred["vert_count"]
                                - gt["intersection"]
                            )
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred["void_intersection"]
                            for gt in pred["matched_gt"]:
                                # group?
                                if gt["instance_id"] < 1000:
                                    num_ignore += gt["intersection"]
                                # small ground truth instances
                                if (
                                    gt["vert_count"] < min_region_size
                                    or gt["med_dist"] > distance_thresh
                                    or gt["dist_conf"] < distance_conf
                                ):
                                    num_ignore += gt["intersection"]
                            proportion_ignore = (
                                float(num_ignore) / pred["vert_count"]
                            )
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first

                    # sorting and cumsum
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(
                        y_score_sorted, return_index=True
                    )
                    num_prec_recall = len(unique_indices) + 1

                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    # https://github.com/ScanNet/ScanNet/pull/26
                    # all predictions are non-matched but also all of them are ignored and not counted as FP
                    # y_true_sorted_cumsum is empty
                    # num_true_examples = y_true_sorted_cumsum[-1]
                    num_true_examples = (
                        y_true_sorted_cumsum[-1]
                        if len(y_true_sorted_cumsum) > 0
                        else 0
                    )
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r

                    # first point in curve is artificial
                    precision[-1] = 1.0
                    recall[-1] = 0.0

                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(
                        recall_for_conv[0], recall_for_conv
                    )
                    recall_for_conv = np.append(recall_for_conv, 0.0)

                    stepWidths = np.convolve(
                        recall_for_conv, [-0.5, 0, 0.5], "valid"
                    )
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)

                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float("nan")
                ap[di, li, oi] = ap_current
    
    # 调试：打印每个类别的GT和预测统计
    print("\n" + "="*80)
    print("PER-CLASS GT AND PREDICTION STATISTICS:")
    print(f"{'Class':<20} {'GT Instances':>12} {'Pred Instances':>15} {'Status':>15}")
    print("-"*80)
    for li, label_name in enumerate(CLASS_LABELS):
        total_gt = 0
        total_pred = 0
        for m in matches:
            total_gt += len(matches[m]["gt"][label_name])
            total_pred += len(matches[m]["pred"][label_name])
        
        if total_gt > 0 or total_pred > 0:
            status = "OK" if total_gt > 0 and total_pred > 0 else ("No Pred" if total_gt > 0 else "No GT")
            print(f"{label_name:<20} {total_gt:>12} {total_pred:>15} {status:>15}")
    print("="*80 + "\n")
    
    return ap


def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(opt["overlaps"], 0.5))
    o25 = np.where(np.isclose(opt["overlaps"], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt["overlaps"], 0.25)))
    avg_dict = {}
    # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])
    avg_dict["all_ap"] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict["all_ap_50%"] = np.nanmean(aps[d_inf, :, o50])
    avg_dict["all_ap_25%"] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["classes"] = {}
    for (li, label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name] = {}
        # avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"] = np.average(
            aps[d_inf, li, oAllBut25]
        )
        avg_dict["classes"][label_name]["ap50%"] = np.average(
            aps[d_inf, li, o50]
        )
        avg_dict["classes"][label_name]["ap25%"] = np.average(
            aps[d_inf, li, o25]
        )
    return avg_dict


def make_pred_info(pred: dict):
    # pred = {'pred_scores' = 100, 'pred_classes' = 100 'pred_masks' = Nx100}
    pred_info = {}
    assert (
        pred["pred_classes"].shape[0]
        == pred["pred_scores"].shape[0]
        == pred["pred_masks"].shape[1]
    )
    for i in range(len(pred["pred_classes"])):
        info = {}
        info["label_id"] = pred["pred_classes"][i]
        info["conf"] = pred["pred_scores"][i]
        info["mask"] = pred["pred_masks"][:, i]
        pred_info[uuid4()] = info  # we later need to identify these objects
    return pred_info


def assign_instances_for_scan(pred: dict, gt_file: str = None, gt_ids = None, scene_id: str = None, stats: dict = None):
    """
    Associate predicted instances with GT, while collecting basic stats.

    Args:
        pred: dict with pred_masks, pred_classes, pred_scores
        gt_file: path to GT ids txt (optional if gt_ids is provided)
        gt_ids: numpy array of GT ids (sem*1000 + inst + 1)
        scene_id: optional scene identifier for logging
        stats: optional dict with counters {total_pred_instances, skipped_instances, length_mismatches}
    """
    logger = logging.getLogger(__name__)
    scene_name = scene_id if scene_id is not None else Path(gt_file).stem if gt_file else "unknown"
    if stats is None:
        stats = {
            "total_pred_instances": 0,
            "skipped_instances": 0,
            "length_mismatches": [],
        }

    pred_info = make_pred_info(pred)

    if gt_ids is None:
        # gt not provided, load from file
        gt_ids = util_3d.load_ids(gt_file)

    # 调试：检查GT数据
    if len(gt_ids) > 0:
        unique_ids = np.unique(gt_ids)
        unique_sem_ids = np.unique(gt_ids // 1000)
        logger.debug(
            "Scene %s: GT loaded - %d points, %d unique instance IDs, semantic IDs: %s",
            scene_name, len(gt_ids), len(unique_ids), unique_sem_ids[:20]  # 只显示前20个
        )
    else:
        logger.warning("Scene %s: GT is EMPTY! len(gt_ids)=%d", scene_name, len(gt_ids))

    # get gt instances
    gt_instances = util_3d.get_instances(
        gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL
    )
    
    # 调试：检查GT实例数量
    total_gt_instances = sum(len(gt_instances[label]) for label in gt_instances)
    if total_gt_instances == 0:
        logger.warning(
            "Scene %s: No GT instances found! VALID_CLASS_IDS=%s, unique_sem_ids=%s",
            scene_name, VALID_CLASS_IDS[:10], np.unique(gt_ids // 1000)[:10]
        )
    # associate
    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt["matched_pred"] = []
    pred2gt = {label: [] for label in CLASS_LABELS}
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.in1d(gt_ids // 1000, VALID_CLASS_IDS))
    # go thru all prediction masks
    for uuid in pred_info:
        label_id = int(pred_info[uuid]["label_id"])
        conf = pred_info[uuid]["conf"]
        if label_id not in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        # read the mask
        pred_mask = pred_info[uuid]["mask"]

        # 容错机制：检查 mask 长度是否匹配
        if len(pred_mask) != len(gt_ids):
            logger.warning(
                "Scene %s: Pred mask length %d != GT length %d, skipping instance %s",
                scene_name,
                len(pred_mask),
                len(gt_ids),
                uuid,
            )
            stats["skipped_instances"] += 1
            if len(stats["length_mismatches"]) < 20:
                stats["length_mismatches"].append(
                    {
                        "scene": scene_name,
                        "pred_len": len(pred_mask),
                        "gt_len": len(gt_ids),
                        "label": label_name,
                    }
                )
            continue  # 跳过这个预测实例

        stats["total_pred_instances"] += 1

        # convert to binary
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < opt["min_region_sizes"][0]:
            continue  # skip if empty

        pred_instance = {
            "uuid": uuid,
            "pred_id": num_pred_instances,
            "label_id": label_id,
            "vert_count": num,
            "confidence": conf,
            "void_intersection": np.count_nonzero(
                np.logical_and(bool_void, pred_mask)
            ),
        }

        # matched gt instances
        matched_gt = []
        # go thru all gt instances with matching label
        for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(
                np.logical_and(gt_ids == gt_inst["instance_id"], pred_mask)
            )
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy["intersection"] = intersection
                pred_copy["intersection"] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]["matched_pred"].append(pred_copy)
        pred_instance["matched_gt"] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt, stats


def print_results(avgs):
    sep = ""
    col1 = ":"
    lineLen = 64

    print("")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>15}".format("AP") + sep
    line += "{:>15}".format("AP_50%") + sep
    line += "{:>15}".format("AP_25%") + sep
    print(line)
    print("#" * lineLen)

    for (li, label_name) in enumerate(CLASS_LABELS):
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg) + sep
        line += sep + "{:>15.3f}".format(ap_50o) + sep
        line += sep + "{:>15.3f}".format(ap_25o) + sep
        print(line)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]

    print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg) + sep
    line += "{:>15.3f}".format(all_ap_50o) + sep
    line += "{:>15.3f}".format(all_ap_25o) + sep
    print(line)
    print("")


def get_output(avgs):
    output = []
    for i in range(len(VALID_CLASS_IDS)):
        class_name = CLASS_LABELS[i]
        class_id = VALID_CLASS_IDS[i]
        ap = avgs["classes"][class_name]["ap"]
        ap50 = avgs["classes"][class_name]["ap50%"]
        ap25 = avgs["classes"][class_name]["ap25%"]
        output.append([class_name, class_id, ap, ap50, ap25])
    return output

def write_result_file(avgs, filename):
    _SPLITTER = ","
    output = []
    with open(filename, "w") as f:
        f.write(
            _SPLITTER.join(["class", "class id", "ap", "ap50", "ap25"]) + "\n"
        )
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            ap = avgs["classes"][class_name]["ap"]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(
                _SPLITTER.join(
                    [str(x) for x in [class_name, class_id, ap, ap50, ap25]]
                )
                + "\n"
            )
            output.append([class_name, class_id, ap, ap50, ap25])


def evaluate(
    preds: dict, gt_path: str, output_file: str, dataset_name: str = "scannet",
    dataset=None, eval_each_scene=False, no_output=False # dont create output file
):
    global CLASS_LABELS
    global VALID_CLASS_IDS
    global ID_TO_LABEL
    global LABEL_TO_ID
    global opt

    if dataset_name == 'scannetpp':
        # get class labels and IDs from the label DB
        CLASS_LABELS = []
        VALID_CLASS_IDS = []
        LABEL_TO_ID = {}
        ID_TO_LABEL = {}

        # pick for training and val
        for ndx, label_dict in dataset._labels.items():
            CLASS_LABELS.append(label_dict['name'])
            VALID_CLASS_IDS.append(ndx)
        VALID_CLASS_IDS = np.array(VALID_CLASS_IDS, dtype=np.int32)

        # get mappings
        for i in range(len(VALID_CLASS_IDS)):
            LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
            ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]
        
        # 调试：打印类别映射信息
        print("\n" + "="*80)
        print(f"EVALUATION SETUP for {dataset_name}:")
        print(f"Total classes: {len(CLASS_LABELS)}")
        print(f"VALID_CLASS_IDS: {VALID_CLASS_IDS[:20]}...")  # 显示前20个
        print(f"First 10 class mappings:")
        for i in range(min(10, len(CLASS_LABELS))):
            print(f"  {VALID_CLASS_IDS[i]:3d} -> {CLASS_LABELS[i]}")
        print("="*80 + "\n")
    
    # used only for s3dis
    if dataset_name == "s3dis":
        total_true = 0
        total_seen = 0
        NUM_CLASSES = len(VALID_CLASS_IDS)

        true_positive_classes = np.zeros(NUM_CLASSES)
        positive_classes = np.zeros(NUM_CLASSES)
        gt_classes = np.zeros(NUM_CLASSES)

        # precision & recall
        total_gt_ins = np.zeros(NUM_CLASSES)
        at = 0.5
        tpsins = [[] for _ in range(NUM_CLASSES)]
        fpsins = [[] for _ in range(NUM_CLASSES)]
        # mucov and mwcov
        all_mean_cov = [[] for _ in range(NUM_CLASSES)]
        all_mean_weighted_cov = [[] for _ in range(NUM_CLASSES)]

    print("Evaluating", len(preds), "scans...")
    matches = {}
    
    # 添加统计信息
    total_pred_instances = 0
    skipped_instances = 0
    length_mismatches = []

    for sample_ndx, (scene_id, scene_preds) in enumerate(preds.items()):
        # key = scene id = scene0000_00 
        # value = pred_masks: nvtx x 100, pred_classes: 100, pred_scores: 100
        # create the gt file key 
        gt_file = os.path.join(gt_path, scene_id + ".txt")

        # get the GT from the dataset, not from GT file
        if dataset_name in ('scannet', 'scannetpp'):
            # 使用scene_id精确查找对应的dataset索引，避免按枚举顺序造成错配
            try:
                scene_index = dataset.scene_ids.index(scene_id)
            except ValueError:
                # 回退：如果scene_id不在列表中，保持原逻辑但记录警告
                print(f"WARNING: scene_id {scene_id} not found in dataset.scene_ids; falling back to order index {sample_ndx}")
                scene_index = sample_ndx
            # this way the GT is consistent even if the data was changed after loading
            gt_data = dataset.__getitem__(scene_index, return_gt_data=True)
        else:
            gt_data = None
            # gt file should exist
            if not os.path.isfile(gt_file):
                util.print_error(
                    "GT file {} not found".format(gt_file), user_fault=True
                )

        if dataset_name == "s3dis":
            gt_ids = util_3d.load_ids(gt_file)
            gt_sem = (gt_ids // 1000) - 1
            gt_ins = gt_ids - (gt_ids // 1000) * 1000

            # pred_sem = v['pred_classes'] - 1
            pred_sem = np.zeros(v["pred_masks"].shape[0], dtype=np.int)
            # TODO CONTINUE HERE!!!!!!!!!!!!!
            pred_ins = np.zeros(v["pred_masks"].shape[0], dtype=np.int)

            for inst_id in reversed(range(v["pred_masks"].shape[1])):
                point_ids = np.argwhere(v["pred_masks"][:, inst_id] == 1.0)[
                    :, 0
                ]
                pred_ins[point_ids] = inst_id + 1
                pred_sem[point_ids] = v["pred_classes"][inst_id] - 1

            # semantic acc
            total_true += np.sum(pred_sem == gt_sem)
            total_seen += pred_sem.shape[0]

            # TODO PARALLELIZ THIS!!!!!!!
            # pn semantic mIoU
            """
            for j in range(gt_sem.shape[0]):
                gt_l = int(gt_sem[j])
                pred_l = int(pred_sem[j])
                gt_classes[gt_l] += 1
                positive_classes[pred_l] += 1
                true_positive_classes[gt_l] += int(gt_l == pred_l)
            """

            uniq, counts = np.unique(pred_sem, return_counts=True)
            positive_classes[uniq] += counts

            uniq, counts = np.unique(gt_sem, return_counts=True)
            gt_classes[uniq] += counts

            uniq, counts = np.unique(
                gt_sem[pred_sem == gt_sem], return_counts=True
            )
            true_positive_classes[uniq] += counts

            # instance
            un = np.unique(pred_ins)
            pts_in_pred = [[] for _ in range(NUM_CLASSES)]
            for ig, g in enumerate(un):  # each object in prediction
                if g == -1:
                    continue
                tmp = pred_ins == g
                sem_seg_i = int(stats.mode(pred_sem[tmp])[0])
                pts_in_pred[sem_seg_i] += [tmp]

            un = np.unique(gt_ins)
            pts_in_gt = [[] for _ in range(NUM_CLASSES)]
            for ig, g in enumerate(un):
                tmp = gt_ins == g
                sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
                pts_in_gt[sem_seg_i] += [tmp]

            # instance mucov & mwcov
            for i_sem in range(NUM_CLASSES):
                sum_cov = 0
                mean_cov = 0
                mean_weighted_cov = 0
                num_gt_point = 0
                for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                    ovmax = 0.0
                    num_ins_gt_point = np.sum(ins_gt)
                    num_gt_point += num_ins_gt_point
                    for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                        union = ins_pred | ins_gt
                        intersect = ins_pred & ins_gt
                        iou = float(np.sum(intersect)) / np.sum(union)

                        if iou > ovmax:
                            ovmax = iou
                            ipmax = ip

                    sum_cov += ovmax
                    mean_weighted_cov += ovmax * num_ins_gt_point

                if len(pts_in_gt[i_sem]) != 0:
                    mean_cov = sum_cov / len(pts_in_gt[i_sem])
                    all_mean_cov[i_sem].append(mean_cov)

                    mean_weighted_cov /= num_gt_point
                    all_mean_weighted_cov[i_sem].append(mean_weighted_cov)

        if dataset_name == "s3dis":
            # instance precision & recall
            for i_sem in range(NUM_CLASSES):
                tp = [0.0] * len(pts_in_pred[i_sem])
                fp = [0.0] * len(pts_in_pred[i_sem])
                gtflag = np.zeros(len(pts_in_gt[i_sem]))
                total_gt_ins[i_sem] += len(pts_in_gt[i_sem])

                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    ovmax = -1.0

                    for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                        union = ins_pred | ins_gt
                        intersect = ins_pred & ins_gt
                        iou = float(np.sum(intersect)) / np.sum(union)

                        if iou > ovmax:
                            ovmax = iou
                            igmax = ig

                    if ovmax >= at:
                        tp[ip] = 1  # true
                    else:
                        fp[ip] = 1  # false positive

                tpsins[i_sem] += tp
                fpsins[i_sem] += fp

        matches_key = os.path.abspath(gt_file)
        # assign gt to predictions
        scene_stats = {
            "total_pred_instances": 0,
            "skipped_instances": 0,
            "length_mismatches": [],
        }
        # 额外一致性检查：pred与gt长度应相同
        pred_len = scene_preds["pred_masks"].shape[0]
        gt_len = len(gt_data) if gt_data is not None else -1
        if gt_len != -1 and pred_len != gt_len:
            print(f"❌ Length mismatch before assignment: scene={scene_id}, pred_len={pred_len}, gt_len={gt_len}")

        gt2pred, pred2gt, scene_stats = assign_instances_for_scan(
            scene_preds, gt_file, gt_ids=gt_data, scene_id=scene_id, stats=scene_stats
        )
        
        # 调试：统计每个场景的GT和预测
        scene_gt_count = sum(len(gt2pred[label]) for label in gt2pred)
        scene_pred_count = sum(len(pred2gt[label]) for label in pred2gt)
        if scene_gt_count > 0 or scene_pred_count > 0:
            gt_by_class = {label: len(gt2pred[label]) for label in gt2pred if len(gt2pred[label]) > 0}
            pred_by_class = {label: len(pred2gt[label]) for label in pred2gt if len(pred2gt[label]) > 0}
            print(f"\n[Scene {sample_ndx+1}/{len(preds)}] {scene_id}:")
            print(f"  GT instances: {scene_gt_count} across {len(gt_by_class)} classes")
            print(f"  Pred instances: {scene_pred_count} across {len(pred_by_class)} classes")
            if scene_gt_count > 0:
                print(f"  GT classes: {list(gt_by_class.keys())[:5]}...")  # 显示前5个
            if scene_pred_count > 0:
                print(f"  Pred classes: {list(pred_by_class.keys())[:5]}...")  # 显示前5个
        
        total_pred_instances += scene_stats["total_pred_instances"]
        skipped_instances += scene_stats["skipped_instances"]
        length_mismatches.extend(scene_stats["length_mismatches"])
        matches[matches_key] = {}
        matches[matches_key]["gt"] = gt2pred
        matches[matches_key]["pred"] = pred2gt
        sys.stdout.write("\rscans processed: {}".format(sample_ndx + 1))
        sys.stdout.flush()

        # write results for each scene to file along with AP50 in the filename
        if eval_each_scene:
            matches_scene = {
                'matches_key': {
                    "gt": gt2pred,
                    "pred": pred2gt
                }
            }
            ap_scores = evaluate_matches(matches_scene)
            avgs_scene = compute_averages(ap_scores)
            scene_ap50 = avgs_scene["all_ap_50%"]
            scene_output_file = Path(output_file).parent / f"{scene_id}-AP50_{scene_ap50}.txt"
            if not no_output:
                write_result_file(avgs_scene, scene_output_file)
    
    # 打印统计信息
    print("\n" + "="*80)
    print("EVALUATION STATISTICS:")
    print(f"Total predicted instances: {total_pred_instances + skipped_instances}")
    print(f"Successfully evaluated instances: {total_pred_instances}")
    print(f"Skipped instances (length mismatch): {skipped_instances}")
    if total_pred_instances + skipped_instances > 0:
        skip_ratio = skipped_instances / (total_pred_instances + skipped_instances) * 100
        print(f"Skip ratio: {skip_ratio:.2f}%")
        if skip_ratio > 5:
            print("⚠️  WARNING: More than 5% of instances were skipped!")
        if skip_ratio > 20:
            print("❌ CRITICAL: More than 20% of instances were skipped! Data may be corrupted.")
    
    if length_mismatches:
        print(f"\nFirst {len(length_mismatches)} length mismatches:")
        for i, mm in enumerate(length_mismatches, 1):
            print(f"  {i}. Scene: {mm['scene']}, Label: {mm['label']}, "
                  f"Pred: {mm['pred_len']}, GT: {mm['gt_len']}, "
                  f"Diff: {mm['pred_len'] - mm['gt_len']}")
    print("="*80 + "\n")
    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)

    # print
    print_results(avgs)
    # get output in a nice list format
    avgs_output = get_output(avgs) 

    if not no_output: 
        write_result_file(avgs, output_file)

    if dataset_name == "s3dis":
        MUCov = np.zeros(NUM_CLASSES)
        MWCov = np.zeros(NUM_CLASSES)
        for i_sem in range(NUM_CLASSES):
            MUCov[i_sem] = np.mean(all_mean_cov[i_sem])
            MWCov[i_sem] = np.mean(all_mean_weighted_cov[i_sem])

        precision = np.zeros(NUM_CLASSES)
        recall = np.zeros(NUM_CLASSES)
        for i_sem in range(NUM_CLASSES):
            tp = np.asarray(tpsins[i_sem]).astype(np.float)
            fp = np.asarray(fpsins[i_sem]).astype(np.float)
            tp = np.sum(tp)
            fp = np.sum(fp)
            rec = tp / total_gt_ins[i_sem]
            prec = tp / (tp + fp)

            precision[i_sem] = prec
            recall[i_sem] = rec

        return np.mean(precision), np.mean(recall)
    
    return avgs_output
