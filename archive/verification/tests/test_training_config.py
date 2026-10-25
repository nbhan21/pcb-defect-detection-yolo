"""Pin validation-only selection and honest detection scoring behavior."""
import unittest
from pcb_inspector.training import match_detections, select_confidence, verify_manifest


class TrainingIntegrityTests(unittest.TestCase):
    def test_duplicate_prediction_is_false_positive(self):
        truth = [{'class_id': 0, 'bbox_xyxy': [0,0,10,10]}]
        predictions = [dict(truth[0], confidence=.9), dict(truth[0], confidence=.8)]
        self.assertEqual(match_detections(predictions, truth), (1,1,0))

    def test_wrong_class_and_bad_location_do_not_match(self):
        truth = [{'class_id': 0, 'bbox_xyxy': [0,0,10,10]}]
        predictions = [{'class_id': 1, 'bbox_xyxy': [0,0,10,10], 'confidence': .9}, {'class_id': 0, 'bbox_xyxy': [20,20,30,30], 'confidence': .8}]
        self.assertEqual(match_detections(predictions, truth), (0,2,1))

    def test_empty_predictions_count_missed_truths(self):
        self.assertEqual(match_detections([], [{'class_id': 0, 'bbox_xyxy':[0,0,10,10]}]), (0,0,1))

    def test_threshold_selection_prefers_f1_then_recall(self):
        metrics = [{'confidence':.2,'precision':.5,'recall':1.0}, {'confidence':.5,'precision':1.0,'recall':.5}]
        self.assertEqual(select_confidence(metrics)['confidence'], .2)

    def test_test_split_cannot_select_confidence(self):
        with self.assertRaises(ValueError):
            select_confidence([{'confidence':.25,'precision':1,'recall':1}], split='test')

    def test_manifest_rejects_source_family_and_hash_leakage(self):
        valid = {'records':[{'family':'a','split':'train','sha256':'1'}, {'family':'b','split':'val','sha256':'2'}, {'family':'c','split':'test','sha256':'3'}]}
        verify_manifest(valid)
        for key in ('family', 'sha256'):
            broken = {'records': [dict(r) for r in valid['records']]}
            broken['records'][1][key] = broken['records'][0][key]
            with self.assertRaises(ValueError):
                verify_manifest(broken)


if __name__ == '__main__':
    unittest.main()
