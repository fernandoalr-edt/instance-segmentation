
class Image(object):
	"""docstring for image"""
	def __init__(self, id_, width, height, file_name, license, date_captured, annotations, new_name):
		super(Image, self).__init__()
		self.id = id_
		self.width = width
		self.height = height
		self.file_name = file_name
		self.license = license
		self.date_captured = date_captured

		self.annotations = annotations

		self.new_name = new_name

	def get_dict(self):

		data_dict = {
			"id": self.id,
			"width":self.width,
			"height":self.height,
			"file_name":self.new_name,
			"license":self.license,
			"date_captured":self.date_captured,
		}

		return data_dict

	def get_annotations_dicts(self):

		annotations_list = []

		for annotation in self.annotations:
			annotations_list.append(annotation.get_dict())

		return annotations_list


class Annotation(object):
	"""docstring for annotation"""
	def __init__(self, segmentation, area, bbox, iscrowd, id_, image_id, category_id):
		super(Annotation, self).__init__()
		self.segmentation = segmentation
		self.area = area
		self.bbox = bbox
		self.iscrowd = iscrowd
		self.id = id_
		self.image_id = image_id
		self.category_id = category_id

	def get_dict(self):

		data_dict = {
			"segmentation":self.segmentation,
			"area":self.area,
			"bbox":self.bbox,
			"iscrowd":self.iscrowd,
			"id":self.id,
			"image_id":self.image_id,
			"category_id":self.category_id
		}

		return data_dict