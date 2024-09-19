import os
import cv2
import json
import random

import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.files import increment_path
from src.data.json_service import load_json, write_json
from src.features.dataset_classes import Image, Annotation


def convert_coco(
	labels_dir="../coco/annotations/",
	images_dir="../coco/images/",
	save_dir="coco_converted/",
	use_segments=False,
	use_keypoints=False,
	cls91to80=True,
	lvis=False,
):
	"""
	Converts COCO dataset annotations to a YOLO annotation format  suitable for training YOLO models.

	Args:
		labels_dir (str, optional): Path to directory containing COCO dataset annotation files.
		save_dir (str, optional): Path to directory to save results to.
		use_segments (bool, optional): Whether to include segmentation masks in the output.
		use_keypoints (bool, optional): Whether to include keypoint annotations in the output.
		cls91to80 (bool, optional): Whether to map 91 COCO class IDs to the corresponding 80 COCO class IDs.
		lvis (bool, optional): Whether to convert data in lvis dataset way.

	Example:
		```python
		from ultralytics.data.converter import convert_coco

		convert_coco("../datasets/coco/annotations/", use_segments=True, use_keypoints=False, cls91to80=True)
		convert_coco("../datasets/lvis/annotations/", use_segments=True, use_keypoints=False, cls91to80=False, lvis=True)
		```

	Output:
		Generates output files in the specified output directory.
	"""
	# Create dataset directory
	save_dir = increment_path(save_dir)  # increment if save directory already exists

	# Import json
	for json_file in sorted(Path(labels_dir).resolve().glob("*.json")):
		lname = "" if lvis else json_file.stem.replace("instances_", "")
		fn = Path(save_dir) / "labels" / lname  # folder name
		with open(json_file) as f:
			data = json.load(f)

		# Create image dict
		images = {f'{x["id"]:d}': x for x in data["images"]}
		# Create image-annotations dict
		imgToAnns = defaultdict(list)
		for ann in data["annotations"]:
			imgToAnns[ann["image_id"]].append(ann)

		image_txt = []
		# Write labels file
		for img_id, anns in TQDM(imgToAnns.items(), desc=f"Annotations {json_file}"):
			img = images[f"{img_id:d}"]
			h, w = img["height"], img["width"]
			f = str(Path(img["coco_url"]).relative_to("http://images.cocodataset.org")) if lvis else os.path.join(images_dir, img["file_name"])
			if lvis:
				image_txt.append(str(Path("./images") / f))

			bboxes = []
			segments = []
			keypoints = []
			for ann in anns:
				if ann.get("iscrowd", False):
					continue
				# The COCO box format is [top left x, top left y, width, height]
				box = np.array(ann["bbox"], dtype=np.float64)
				box[:2] += box[2:] / 2  # xy top-left corner to center
				box[[0, 2]] /= w  # normalize x
				box[[1, 3]] /= h  # normalize y
				if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
					continue

				cls = ann["category_id"]
				box = [cls] + box.tolist()
				if box not in bboxes:
					bboxes.append(box)
					if use_segments and ann.get("segmentation") is not None:
						if len(ann["segmentation"]) == 0:
							segments.append([])
							continue
						elif len(ann["segmentation"]) > 1:
							s = merge_multi_segment(ann["segmentation"])
							s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
						else:
							s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
							s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
						s = [cls] + s
						segments.append(s)
					if use_keypoints and ann.get("keypoints") is not None:
						keypoints.append(
							box + (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
						)

			# Write
			with open((fn / f).with_suffix(".txt"), "w") as file:
				for i in range(len(bboxes)):
					if use_keypoints:
						line = (*(keypoints[i]),)  # cls, box, keypoints
					else:
						line = (
							*(segments[i] if use_segments and len(segments[i]) > 0 else bboxes[i]),
						)  # cls, box or segments
					file.write(("%g " * len(line)).rstrip() % line + "\n")

		if lvis:
			with open((Path(save_dir) / json_file.name.replace("lvis_v1_", "").replace(".json", ".txt")), "a") as f:
				f.writelines(f"{line}\n" for line in image_txt)

	LOGGER.info(f"{'LVIS' if lvis else 'COCO'} data converted successfully.\nResults saved to {save_dir.resolve()}")


def unificate_category(category, data_dir):

	image_index = 0
	annotation_id = 0

	category_data = load_json("src/features/class_mapper.json")[category]

	new_data_dict = {"categories":[{"supercategory":category,
									"id":category_data[key]["Id"],
									"name":category_data[key]["Name"]} for key in category_data],
					 "images":[],
					 "annotations":[]}

	image_save_dir = os.path.join(data_dir, category, "Antiguo", "Images")
	coco_json_save_path = os.path.join(data_dir, category, "Antiguo", "CoCo_Annotations", "coco.json")

	old_ds_path = os.path.join(data_dir, "00-OLD")
	old_ds_list = os.listdir(old_ds_path)

	images_list = []

	for old_ds in tqdm(old_ds_list):
		jsons_list = ["COCO_ANNOTATIONS_TRAIN.json","COCO_ANNOTATIONS_TEST.json","COCO_ANNOTATIONS_VAL.json"]
		for json_file in jsons_list:
			json_path = os.path.join(old_ds_path, old_ds, "CoCo_Annotations", json_file)

			data_dict = load_json(json_path)

			for image in data_dict['images']:
				annotations_found = False
				image_id = image["id"]

				annotations_list = []

				for annotation in data_dict['annotations']:
					if annotation['image_id'] == image_id:
						if str(annotation['category_id']) in list(category_data.keys()):

							annotations_found = True
							annotations_list.append(Annotation(annotation['segmentation'],
															   annotation['area'],
															   annotation['bbox'],
															   annotation['iscrowd'],
															   annotation_id,
															   image_index,
															   category_data[str(annotation["category_id"])]["Id"]))

							annotation_id += 1
				if annotations_found:
					images_list.append(Image(image_index,
											 image['width'],
											 image['height'],
											 os.path.join(old_ds_path, old_ds, 'Images', image['file_name']),
											 0,
											 0,
											 annotations_list,
											 os.path.join(image_save_dir, f"{image_index}.png")))
					image_index += 1

	for image in tqdm(images_list):

		img = cv2.imread(image.file_name)
		if img is None: continue
		cv2.imwrite(image.new_name, img)

		new_data_dict['annotations'] += image.get_annotations_dicts()
		new_data_dict['images'].append(image.get_dict())

	write_json(new_data_dict, coco_json_save_path)


def generate_txt_files(data_dir, category):

	subsets_list = ["train","val","test"]

	own_json_path = os.path.join(data_dir, category, "Propio", "CoCo_Annotations", "coco.json")
	own_images_list = [os.path.join(data_dir, category, "Propio", "Images", image["file_name"]) 
					   for image in load_json(own_json_path)["images"]]

	subsets_sizes = [round(len(own_images_list)*0.7),
					 round(len(own_images_list)*0.1),
					 round(len(own_images_list)*0.2)]

	for subset, size in zip(subsets_list, subsets_sizes):

		ADE20K_json_path = os.path.join(data_dir, category, "ADE20K", f"{subset}.json")
		ADE20K_images_list = [os.path.join(data_dir, "ADE20K", image["file_name"])
							  for image in load_json(ADE20K_json_path)["images"]]

		if subset != "test":
			own_subsample_list = random.sample(own_images_list, size)
		else:
			own_subsample_list = own_images_list

		images_list = own_subsample_list + ADE20K_images_list

		txt_save_path = os.path.join(data_dir, category, subset+".txt")
		with open(txt_save_path, "w") as f:
			for line in images_list:
				f.write(line + '\n')

		own_images_list = [image_path for image_path in own_images_list if image_path not in own_subsample_list]
