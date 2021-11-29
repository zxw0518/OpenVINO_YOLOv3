import sys
#注意paddlex的版本是1.3.11而不是最新版本，否则会有报错，result中标记的类别是按照COCO数据集来的
from openvino_ppdet.yolov3_infer import YOLOv3

xml_file = "ov_model/yolov3.xml"
bin_file = "ov_model/yolov3.bin"
model = YOLOv3(xml_file=xml_file,
               bin_file=bin_file,
               model_input_shape=[608, 608])
boxes = model.predict("images/cd.jpg", visualize_out="images/result_cd.jpg", threshold=0.5)