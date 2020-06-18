import logging
import os
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import tensorflow as tf

from reducto.data_loader import load_yaml
from reducto.video_processor import VideoProcessor


class ObjectDetectionModel:

    def __init__(self):
        self.name = '__generic_model__'

    def infer_video(self, video_path, transformer=None):
        video_name = Path(video_path).stem
        time_start = time.time()
        result = {}
        with VideoProcessor(video_path) as video:
            for frame in video:
                if transformer and not transformer.is_identity():
                    frame = transformer(frame)
                result[video.index] = self._infer_image(frame)
        time_end = time.time()
        # logging.info(f'{video_name},inference,,{time_start},{time_end},{time_end - time_start}')
        return result

    def _infer_image(self, frame):
        raise NotImplementedError()

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DetectionZoo(ObjectDetectionModel):

    def __init__(self, model_name='ssdlite_mobilenet_v2_coco', no_session=False):
        super().__init__()

        self.name = model_name

        self._load_model(model_name, model_root=os.getenv('model_root', './weights'))
        self.image_tensor = self.graph.get_tensor_by_name(f'{model_name}/image_tensor:0')
        self.detection_boxes = self.graph.get_tensor_by_name(f'{model_name}/detection_boxes:0')

        if no_session:
            return

        self.sess = tf.compat.v1.Session(graph=self.graph, config=tf.compat.v1.ConfigProto(
            # log_device_placement=True,
            allow_soft_placement=True,
        ))

    def close(self):
        self.sess.close()

    def _infer_image(self, image):
        # Loads the image
        (im_height, im_width, _) = image.shape
        image = np.expand_dims(image, 0)

        # Prepares input/output tensor dicts
        tensor_names = {output.name for op in self.graph.get_operations() for output in op.outputs}
        tensor_dict = {}
        for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
            tensor_name = f'{self.name}/{key}:0'
            if tensor_name in tensor_names:
                tensor_dict[key] = self.graph.get_tensor_by_name(tensor_name)

        # Runs the inference
        output = self.sess.run(tensor_dict, feed_dict={self.image_tensor: image})
        num_detections = int(output['num_detections'][0])
        return {
            'num_detections': num_detections,
            'detection_boxes': output['detection_boxes'][0][:num_detections].tolist(),
            'detection_scores': output['detection_scores'][0][:num_detections].tolist(),
            'detection_classes': output['detection_classes'][0].astype(np.uint8)[:num_detections].tolist()
        }

    def _load_model(self, model_name, model_root):
        download_base_url = 'http://download.tensorflow.org/models/object_detection/'
        model_list = load_yaml('config/zoo_models.yaml')
        trained_model_name = [model for model in model_list if model['name'] == model_name][0]['trained_name']
        trained_model_filename = f'{trained_model_name}.tar.gz'
        trained_model_url = f'{download_base_url}{trained_model_filename}'

        # Downloads model data if needed
        if not Path(trained_model_filename).exists():
            if not Path(model_root).exists():
                Path(model_root).mkdir(parents=True, exist_ok=True)
            logging.info(f'downloading model {model_name} to {model_root}')
            trained_model_file = requests.get(trained_model_url)
            with open(f'{model_root}/{trained_model_filename}', 'wb') as f:
                f.write(trained_model_file.content)
            tarfile.open(f'{model_root}/{trained_model_filename}').extractall(f'{model_root}/')

        # Loads frozen graph
        frozen_graph_path = f'{model_root}/{trained_model_name}/frozen_inference_graph.pb'
        with tf.io.gfile.GFile(frozen_graph_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name=model_name)
            self.graph = graph
        logging.info(f'Model({self.name}) successfully loaded')


class Yolo(ObjectDetectionModel):

    def __init__(self, resized=None, no_session=False, _load_weights=True, gpu_id=0):
        super().__init__()
        self.name = 'yolo_v3'
        self.class_num = 80
        self.resized = resized or [624, 624]
        self.anchors = np.array([[10., 13.], [16., 30.], [33., 23.], [30., 61.],
                                 [62., 45.], [59., 119.], [116., 90.],
                                 [156., 198.], [373., 326.]], dtype=np.float32)
        self.use_label_smooth = False
        self.use_focal_loss = False
        self.batch_norm_decay = 0.999
        self.weight_decay = 5e-4
        self.use_static_shape = True

        if no_session:
            return

        self.sess = tf.compat.v1.Session()

        if _load_weights:
            self._load_weights()

    def _infer_image(self, frame):

        im_height, im_width = frame.shape[:2]
        frame = cv2.resize(frame, tuple(self.resized))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # <- ##############
        frame = np.asarray(frame, np.float32)
        frame = frame[np.newaxis, :] / 255.

        boxes, scores, classes = self.sess.run(self.nms, feed_dict={self.input_data: frame})
        boxes[:, [0, 2]] *= (im_width / float(self.resized[0]))
        boxes[:, [1, 3]] *= (im_height / float(self.resized[1]))
        return {
            'num_detections': len(boxes),
            'detection_boxes': boxes.tolist(),
            'detection_scores': scores.tolist(),
            'detection_classes': classes.tolist()
        }

    @staticmethod
    def download_weights(weight_root=f'{os.getenv("model_root", "weights")}/yolov3'):
        weight_url = 'https://pjreddie.com/media/files/yolov3.weights'
        weight_orig_file = f'{weight_root}/yolov3.weights'
        weight_ckpt_file = f'{weight_root}/yolov3.ckpt'

        if not Path(weight_root).exists():
            Path(weight_root).mkdir(parents=True)

        # Downloads model weights if not exists
        if not Path(weight_orig_file).exists():
            logging.info(f'downloading YOLOv3 weights')
            weights = requests.get(weight_url)
            with open(weight_orig_file, 'wb') as f:
                f.write(weights.content)

        # Converts to tensorflow graph if not exists
        resized = 624
        model = Yolo(_load_weights=False)
        with tf.compat.v1.Session() as sess:
            logging.info(f'converting YOLOv3 weights to ckpt')
            inputs = tf.compat.v1.placeholder(tf.float32, [1, resized, resized, 3])
            with tf.compat.v1.variable_scope('yolov3'):
                model.forward(inputs)
            saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(scope='yolov3'))
            load_ops = model.load_weights(tf.compat.v1.global_variables(scope='yolov3'), weight_orig_file)
            sess.run(load_ops)
            saver.save(sess, save_path=weight_ckpt_file)

    def _load_weights(self, weight_root=f'{os.getenv("model_root", "weights")}/yolov3'):
        weight_ckpt_file = f'{weight_root}/yolov3.ckpt'
        # Loads weights
        self.input_data = tf.compat.v1.placeholder(
            tf.float32, [1, self.resized[1], self.resized[0], 3], name='input_data')
        with tf.compat.v1.variable_scope('yolov3'):
            pred_feature_maps = self.forward(self.input_data)
        pred_boxes, pred_confs, pred_probs = self.predict(pred_feature_maps)
        pred_scores = pred_confs * pred_probs
        boxes, scores, labels = Yolo.gpu_nms(pred_boxes, pred_scores)
        self.nms = [boxes, scores, labels]
        self.restore_path = weight_ckpt_file
        tf.compat.v1.train.Saver().restore(self.sess, self.restore_path)

    @staticmethod
    def load_weights(var_list, weights_file):
        """
        Loads and converts pre-trained weights.
        param:
            var_list: list of network variables.
            weights_file: name of the binary file.
        """
        with open(weights_file, 'rb') as fp:
            np.fromfile(fp, dtype=np.int32, count=5)
            weights = np.fromfile(fp, dtype=np.float32)

        ptr, i, assign_ops = 0, 0, []
        while i < len(var_list) - 1:
            var1, var2 = var_list[i], var_list[i+1]
            # do something only if we process conv layer
            if 'Conv' in var1.name.split('/')[-2]:
                # check type of next layer
                if 'BatchNorm' in var2.name.split('/')[-2]:
                    # load batch norm params
                    gamma, beta, mean, var = var_list[i + 1:i + 5]
                    batch_norm_vars = [beta, gamma, mean, var]
                    for var in batch_norm_vars:
                        shape = var.shape.as_list()
                        num_params = np.prod(shape)
                        var_weights = weights[ptr:ptr + num_params].reshape(shape)
                        ptr += num_params
                        assign_ops.append(tf.compat.v1.assign(var, var_weights, validate_shape=True))
                    # we move the pointer by 4, because we loaded 4 variables
                    i += 4
                elif 'Conv' in var2.name.split('/')[-2]:
                    # load biases
                    bias = var2
                    bias_shape = bias.shape.as_list()
                    bias_params = np.prod(bias_shape)
                    bias_weights = weights[ptr:ptr +
                                               bias_params].reshape(bias_shape)
                    ptr += bias_params
                    assign_ops.append(tf.compat.v1.assign(bias, bias_weights, validate_shape=True))
                    # we loaded 1 variable
                    i += 1
                # we can load weights of conv layer
                shape = var1.shape.as_list()
                num_params = np.prod(shape)

                var_weights = weights[ptr:ptr + num_params].reshape(
                    (shape[3], shape[2], shape[0], shape[1]))
                # remember to transpose to column-major
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                assign_ops.append(
                    tf.compat.v1.assign(var1, var_weights, validate_shape=True))
                i += 1

        return assign_ops

    @staticmethod
    def get_color_table(class_num, seed=2):
        import random
        random.seed(seed)
        color_table = {}
        for i in range(class_num):
            color_table[i] = [random.randint(0, 255) for _ in range(3)]
        return color_table

    @staticmethod
    def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
        """
        coord: [x_min, y_min, x_max, y_max] format coordinates.
        img: img to plot on.
        label: str. The label name.
        color: int. color index.
        line_thickness: int. rectangle line thickness.
        """
        tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
        import random
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    def test_infer_video_with_frame_output(self, video_path):
        """test_function"""

        vid = cv2.VideoCapture(video_path)
        video_frame_cnt = int(vid.get(7))
        video_width = int(vid.get(3))
        video_height = int(vid.get(4))
        video_fps = int(vid.get(5))

        print(video_frame_cnt, video_width, video_height, video_fps)

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))

        coco_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 4: 'aeroplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
            10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
            13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
            18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
            23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon',
            45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
            50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut',
            55: 'cake', 56: 'chair', 57: 'sofa', 58: 'pottedplant', 59: 'bed',
            60: 'diningtable', 61: 'toilet', 62: 'tvmonitor', 63: 'laptop',
            64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
            68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
            76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
        color_table = self.get_color_table(80)

        for i in range(video_frame_cnt):
            ret, img_ori = vid.read()
            print(ret, img_ori)
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(self.resized))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.

            boxes_, scores_, labels_ = self.sess.run(self.nms, feed_dict={self.input_data: img})

            boxes_[:, [0, 2]] *= (width_ori/float(self.resized[0]))
            boxes_[:, [1, 3]] *= (height_ori/float(self.resized[1]))

            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                self.plot_one_box(img_ori, [x0, y0, x1, y1],
                                  label=coco_classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                                  color=color_table[labels_[i]])
            cv2.putText(img_ori, '{:.2f}ms'.format(1 * 1000), (40, 40), 0,
                        fontScale=1, color=(0, 255, 0), thickness=2)
            # cv2.imshow('image', img_ori)
            # videoWriter.write(img_ori)
            cv2.imwrite(f'imges/{i}.jpg', img_ori)

    def forward(self, inputs, is_training=False, reuse=False):
        slim = tf.contrib.slim
        # the input img_size, form: [height, weight], will be used later
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                with tf.compat.v1.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = Yolo.darknet53_body(inputs)
                with tf.compat.v1.variable_scope('yolov3_head'):
                    inter1, net = Yolo.yolo_block(route_3, 512)
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1, stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')
                    inter1 = Yolo.conv2d(inter1, 256, 1)
                    inter1 = Yolo.upsample_layer(inter1, route_2.get_shape().as_list()
                    if self.use_static_shape else tf.shape(route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)
                    inter2, net = Yolo.yolo_block(concat1, 256)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1, stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')
                    inter2 = Yolo.conv2d(inter2, 128, 1)
                    inter2 = Yolo.upsample_layer(inter2, route_1.get_shape().as_list()
                    if self.use_static_shape else tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)
                    _, feature_map_3 = Yolo.yolo_block(concat2, 128)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1, stride=1,
                                                normalizer_fn=None, activation_fn=None,
                                                biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

    def predict(self, feature_maps):
        """
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        """
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = x_y_offset.get_shape().as_list()[:2] if self.use_static_shape else tf.shape(x_y_offset)[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs

    def reorg_layer(self, feature_map, anchors):
        '''
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
        anchors: shape: [3, 2]
        '''
        # NOTE: size in [h, w] format! don't get messed up!
        grid_size = feature_map.get_shape().as_list()[1:3] if self.use_static_shape else tf.shape(feature_map)[
                                                                                         1:3]  # [13, 13]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format!
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)

        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2]
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        # get the absolute box coordinates on the feature_map
        box_centers = box_centers + x_y_offset
        # rescale to the original image scale
        box_centers = box_centers * ratio[::-1]

        # avoid getting possible nan value with tf.clip_by_value
        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        # box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 100) * rescaled_anchors
        # rescale to the original image scale
        box_sizes = box_sizes * ratio[::-1]

        # shape: [N, 13, 13, 3, 4]
        # last dimension: (center_x, center_y, w, h)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits

    def loss_layer(self, feature_map_i, y_true, anchors):
        '''
        calc loss function from a certain scale
        input:
            feature_map_i: feature maps of a certain scale. shape: [N, 13, 13, 3*(5 + num_class)] etc.
            y_true: y_ture from a certain scale. shape: [N, 13, 13, 3, 5 + num_class + 1] etc.
            anchors: shape [9, 2]
        '''

        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)

        ###########
        # get mask
        ###########

        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]
        object_mask = y_true[..., 4:5]

        # the calculation of ignore mask if referred from
        # https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c#L179
        ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        def loop_cond(idx, ignore_mask):
            return tf.less(idx, tf.cast(N, tf.int32))

        def loop_body(idx, ignore_mask):
            # shape: [13, 13, 3, 4] & [13, 13, 3]  ==>  [V, 4]
            # V: num of true gt box of each image in a batch
            valid_true_boxes = tf.boolean_mask(y_true[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))
            # shape: [13, 13, 3, 4] & [V, 4] ==> [13, 13, 3, V]
            iou = Yolo.box_iou(pred_boxes[idx], valid_true_boxes)
            # shape: [13, 13, 3]
            best_iou = tf.reduce_max(iou, axis=-1)
            # shape: [13, 13, 3]
            ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
            # finally will be shape: [N, 13, 13, 3]
            ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
            return idx + 1, ignore_mask

        _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # get_tw_th
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment:
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (
                    y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        ############
        # loss_part
        ############
        # mix_up weight
        # [N, 13, 13, 3, 1]
        mix_w = y_true[..., -1:]
        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        # TODO: may need to balance the pos-neg by multiplying some weights
        conf_loss = conf_loss_pos + conf_loss_neg
        if self.use_focal_loss:
            alpha = 1.0
            gamma = 2.0
            # TODO: alpha should be a mask array if needed
            focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
            conf_loss *= focal_mask
        conf_loss = tf.reduce_sum(conf_loss * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        # whether to use label smooth
        if self.use_label_smooth:
            delta = 0.01
            label_target = (1 - delta) * y_true[..., 5:-1] + delta * 1. / self.class_num
        else:
            label_target = y_true[..., 5:-1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target,
                                                                           logits=pred_prob_logits) * mix_w
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    def compute_loss(self, y_pred, y_true):
        """
        param:
            y_pred: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
            y_true: input y_true by the tf.data pipeline
        """
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        # calc loss in 3 scales
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

    @staticmethod
    def box_iou(pred_boxes, valid_true_boxes):
        """
        param:
            pred_boxes: [13, 13, 3, 4], (center_x, center_y, w, h)
            valid_true: [V, 4]
        """
        # [13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]
        # shape: [13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)
        # [V, 2]
        true_box_xy = valid_true_boxes[:, 0:2]
        true_box_wh = valid_true_boxes[:, 2:4]
        # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
        # shape: [13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = tf.expand_dims(true_box_area, axis=0)
        # [13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)
        return iou

    @staticmethod
    def conv2d(inputs, filters, kernel_size, strides=1):

        def _fixed_padding(inputs, kernel_size):
            pad_total = kernel_size - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                            [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
            return padded_inputs

        if strides > 1:
            inputs = _fixed_padding(inputs, kernel_size)
        inputs = tf.contrib.slim.conv2d(inputs, filters, kernel_size, stride=strides,
                                        padding=('SAME' if strides == 1 else 'VALID'))
        return inputs

    @staticmethod
    def darknet53_body(inputs):

        def res_block(inputs, filters):
            shortcut = inputs
            net = Yolo.conv2d(inputs, filters * 1, 1)
            net = Yolo.conv2d(net, filters * 2, 3)
            net = net + shortcut
            return net

        # first two conv2d layers
        net = Yolo.conv2d(inputs, 32, 3, strides=1)
        net = Yolo.conv2d(net, 64, 3, strides=2)

        # res_block * 1
        net = res_block(net, 32)

        net = Yolo.conv2d(net, 128, 3, strides=2)

        # res_block * 2
        for i in range(2):
            net = res_block(net, 64)

        net = Yolo.conv2d(net, 256, 3, strides=2)

        # res_block * 8
        for i in range(8):
            net = res_block(net, 128)

        route_1 = net
        net = Yolo.conv2d(net, 512, 3, strides=2)

        # res_block * 8
        for i in range(8):
            net = res_block(net, 256)

        route_2 = net
        net = Yolo.conv2d(net, 1024, 3, strides=2)

        # res_block * 4
        for i in range(4):
            net = res_block(net, 512)
        route_3 = net

        return route_1, route_2, route_3

    @staticmethod
    def yolo_block(inputs, filters):
        net = Yolo.conv2d(inputs, filters * 1, 1)
        net = Yolo.conv2d(net, filters * 2, 3)
        net = Yolo.conv2d(net, filters * 1, 1)
        net = Yolo.conv2d(net, filters * 2, 3)
        net = Yolo.conv2d(net, filters * 1, 1)
        route = net
        net = Yolo.conv2d(net, filters * 2, 3)
        return route, net

    @staticmethod
    def upsample_layer(inputs, out_shape):
        new_height, new_width = out_shape[1], out_shape[2]
        inputs = tf.compat.v1.image.resize_nearest_neighbor(
            inputs, (new_height, new_width), name='upsampled')
        return inputs

    @staticmethod
    def gpu_nms(_boxes, _scores, num_classes=80, max_boxes=200, score_thresh=0.3, nms_thresh=0.45):
        """
        Perform NMS on GPU using TensorFlow.
        params:
            boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
            scores: tensor of shape [1, 10647, num_classes], score=conf*prob
            num_classes: total number of classes
            max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
            score_thresh: if [ highest class probability score < score_threshold]
                            then get rid of the corresponding box
            nms_thresh: real value, "intersection over union" threshold used for NMS filtering
        """

        boxes_list, label_list, score_list = [], [], []
        max_boxes = tf.constant(max_boxes, dtype='int32')

        # since we do nms for single image, then reshape it
        _boxes = tf.reshape(_boxes, [-1, 4])  # '-1' means we don't know the exact number of boxes
        _score = tf.reshape(_scores, [-1, num_classes])

        # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
        mask = tf.greater_equal(_score, tf.constant(score_thresh))
        # Step 2: Do non_max_suppression for each class
        for i in range(num_classes):
            # Step 3: Apply the mask to scores, boxes and pick them out
            filter_boxes = tf.boolean_mask(_boxes, mask[:, i])
            filter_score = tf.boolean_mask(_score[:, i], mask[:, i])
            nms_indices = tf.image.non_max_suppression(
                boxes=filter_boxes, scores=filter_score,
                max_output_size=max_boxes, iou_threshold=nms_thresh, name='nms_indices')
            label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32') * i)
            boxes_list.append(tf.gather(filter_boxes, nms_indices))
            score_list.append(tf.gather(filter_score, nms_indices))

        _boxes = tf.concat(boxes_list, axis=0)
        _score = tf.concat(score_list, axis=0)
        _label = tf.concat(label_list, axis=0)
        return _boxes, _score, _label
