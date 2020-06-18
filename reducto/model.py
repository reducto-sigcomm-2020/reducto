from mongoengine import *


class Segment(Document):

    subset = StringField(required=True)
    name = StringField(required=True)

    num_frames = IntField(default=-1)
    width = IntField(default=-1)
    height = IntField(default=-1)
    size = IntField(default=-1)

    meta = {
        'indexes': [{
            'fields': ('subset', 'name'),
            'unique': True
        }]
    }

    @staticmethod
    def find_or_save(subset, name):
        record = Segment.objects(
            subset=subset,
            name=name,
        ).first()
        if record:
            return record
        record = Segment(
            subset=subset,
            name=name,
        )
        record.save()
        return record


class InferenceResult(EmbeddedDocument):
    num_detections = IntField()
    detection_scores = ListField()
    detection_classes = ListField()
    detection_boxes = ListField()

    @staticmethod
    def from_json(json_data):
        return InferenceResult(
            num_detections=json_data['num_detections'],
            detection_scores=json_data['detection_scores'],
            detection_classes=json_data['detection_classes'],
            detection_boxes=json_data['detection_boxes'],
        )

    def to_json(self):
        return {
            'num_detections': self.num_detections,
            'detection_scores': self.detection_scores,
            'detection_classes': self.detection_classes,
            'detection_boxes': self.detection_boxes,
        }


class Inference(Document):
    segment = ReferenceField(Segment, required=True)
    model = StringField(required=True)
    result = ListField(EmbeddedDocumentField(InferenceResult), required=True)

    meta = {
        'indexes': [{
            'fields': ('segment', 'model'),
            'unique': True
        }]
    }

    def to_json(self):
        return {
            i + 1: self.result[i].to_json()
            for i in range(len(self.result))
        }


class DiffVector(Document):
    segment = ReferenceField(Segment, required=True)
    differencer = StringField(required=True)
    vector = ListField(required=True)

    meta = {
        'indexes': [{
            'fields': ('segment', 'differencer'),
            'unique': True
        }]
    }


class MotionVector(Document):
    segment = ReferenceField(Segment, required=True)
    motioner = StringField(required=True)
    vector = ListField(required=True)

    meta = {
        'indexes': [{
            'fields': ('segment', 'motioner'),
            'unique': True
        }]
    }


class FrameEvaluation(Document):
    segment = ReferenceField(Segment, required=True)
    model = StringField(required=True)
    evaluator = StringField(required=True)
    ground_truth = IntField(required=True)
    comparision = IntField(required=True)
    result = FloatField(required=True)

    meta = {
        'indexes': [{
            'fields': ('segment', 'model', 'evaluator', 'ground_truth', 'comparision'),
            'unique': True
        }]
    }
