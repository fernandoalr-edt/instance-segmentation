import pandas as pd
import numpy as np
import torch
import cv2

def get_IoU(true_bbox, pred_bbox):
	xA = max(true_bbox[0], pred_bbox[0])
	yA = max(true_bbox[1], pred_bbox[1])
	xB = min(true_bbox[2], pred_bbox[2])
	yB = min(true_bbox[3], pred_bbox[3])

	intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	true_box_area = (true_bbox[2] - true_bbox[0] + 1) * (true_bbox[3] - true_bbox[1] + 1)
	pred_box_area = (pred_bbox[2] - pred_bbox[0] + 1) * (pred_bbox[3] - pred_bbox[1] + 1)

	IoU = intersection_area / float(true_box_area + pred_box_area - intersection_area)

	return IoU


def conf_matrix_calc(predictions_compare_list, classes_names, IoU_threshold=0.5):

	print('*********Generando métricas*********')
	metrics = {}
	for class_name in classes_names:
		metrics[class_name] = {
			"TP":0,
			"TN":0,
			"FP":0,
			"FN":0
		}

	for prediction_compare in predictions_compare_list:

		for pred_class, pred_bbox in zip(prediction_compare.pred_classes, prediction_compare.pred_bboxes):
			found = False
			for true_class, true_bbox in zip(prediction_compare.true_classes, prediction_compare.true_bboxes):
				#if get_IoU(true_bbox, pred_bbox) >= IoU_threshold:
				if true_class == pred_class:
					metrics[classes_names[pred_class]]["TP"] += 1
					found = True
					break

			if not found:
				metrics[classes_names[pred_class]]["FP"] += 1

		non_detected_classes = [class_id for class_id in prediction_compare.true_classes if class_id not in prediction_compare.pred_classes]
		for non_detected_class in non_detected_classes:
			metrics[classes_names[non_detected_class]]["FN"] += 1

	return metrics
