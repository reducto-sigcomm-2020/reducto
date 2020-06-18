from reducto.data_loader import load_json
from reducto.differencer import DiffProcessor


class DiffComposer:

    def __init__(self, differ_dict=None):
        self.differ_dict = differ_dict

    @staticmethod
    def from_jsonfile(jsonpath, differencer_types=None):
        differencer_types = differencer_types or ['pixel', 'area', 'corner', 'edge']
        differ_dict = load_json(jsonpath)
        differencers = {
            feature: threshes
            for feature, threshes in differ_dict.items()
            if feature in differencer_types
        }
        return DiffComposer(differencers)

    @staticmethod
    def placeholder(differencer_types=None):
        differencer_types = differencer_types or ['pixel', 'area', 'corner', 'edge']
        differencers = {
            feature: 0
            for feature in differencer_types
        }
        return DiffComposer(differencers)

    def new_thresholds(self, thresholds):
        for dp, threshes in thresholds.items():
            self.differ_dict[dp] = threshes

    def process_video(self, filepath, diff_vectors=None):
        if diff_vectors:
            assert all([k in diff_vectors for k in self.differ_dict.keys()]), \
                'not compatible diff-vector list'
        else:
            diff_vectors = {
                k: self.get_diff_vector(k, filepath)
                for k in self.differ_dict.keys()
            }

        results = {}
        for differ_type, thresholds in self.differ_dict.items():
            diff_vector = diff_vectors[differ_type]
            result = self.batch_diff(diff_vector, thresholds)
            results[differ_type] = {
                'diff_vector': diff_vector,
                'result': result,
            }
        return results

    @staticmethod
    def get_diff_vector(differ_type, filepath):
        differ = DiffProcessor.str2class(differ_type)()
        return differ.get_diff_vector(filepath)

    @staticmethod
    def batch_diff(diff_vector, thresholds):
        result = DiffProcessor.batch_diff_noobj(diff_vector, thresholds)
        return result
