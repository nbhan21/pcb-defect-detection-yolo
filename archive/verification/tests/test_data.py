"""Synthetic fixtures test data integrity, not PCB model performance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from PIL import Image

from pcb_inspector.data import (
    family_from_name, parse_label, resolve_label, split_families, prepare_dataset,
)


class DataIntegrityTests(unittest.TestCase):
    def test_archive_bytes_are_the_same_bytes_used_for_prepared_images(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            raw = base / 'raw'
            images, labels = raw / 'train/images', raw / 'train/labels'
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            with ZipFile(base / 'pcb-defect-dataset.zip', 'w') as archive:
                for i in range(10):
                    stem = f'light_{i:02}_spur_01_1_600'
                    image = images / (stem + '.png')
                    label = labels / (stem + '.txt')
                    Image.new('RGB', (32, 32), (i*20, 30, 70)).save(image)
                    label.write_text('\n'.join(f'{c} .5 .5 .2 .2' for c in range(6)))
                    for path in (image, label):
                        archive.write(path, 'raw/' + path.relative_to(raw).as_posix())
                    Image.new('RGB', (32, 32), 'white').save(image)
            prepare_dataset(raw, base / 'prepared', base / 'reports')
            manifest = json.loads((base / 'prepared/manifest.json').read_text())
            for record in manifest['records']:
                output = base / 'prepared' / record['split'] / 'images' / record['filename']
                self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), record['sha256'])

    def test_family_groups_rotations_and_prefix_variants(self):
        for name in ('light_01_missing_hole_01_1_600', 'l_light_01_spur_04_2_600', 'rotation_270_light_01_short_01_2_600'):
            self.assertEqual(family_from_name(name), 'light_01')
        with self.assertRaises(ValueError):
            family_from_name('unidentified-board')

    def test_resolve_label_repairs_suffix_only_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as temp:
            labels = Path(temp)
            old = labels / 'light_01_spur_01_1_256.txt'
            old.write_text('1 .5 .5 .2 .2')
            image = Path('light_01_spur_01_1_600.jpg')
            self.assertEqual(resolve_label(image, labels), old)
            (labels / (image.stem + '.txt')).write_text('1 .5 .5 .2 .2')
            with self.assertRaises(ValueError):
                resolve_label(image, labels)

    def test_missing_label_is_a_hard_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                resolve_label(Path('light_01_spur_01_1_600.jpg'), Path(temp))

    def test_parse_rejects_bad_class_geometry_and_nonfinite_values(self):
        for text in ('6 .5 .5 .2 .2', '0 nan .5 .2 .2', '0 .9 .5 .4 .2', '0 .5 .5 0 .2', '0 .5 .5 .2', '0.5 .5 .5 .2 .2'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_label(text)
        self.assertEqual(parse_label(''), [])
        self.assertEqual(parse_label('0 .5 .5 .2 .2')[0][0], 0)

    def test_family_split_is_deterministic_and_disjoint(self):
        records = [{'family': f'light_{i:02}', 'sha256': str(i), 'classes': list(range(6))} for i in range(10)]
        split = split_families(records)
        self.assertEqual(split, split_families(list(reversed(records))))
        self.assertEqual(sorted(split.values()).count('train'), 6)
        self.assertEqual(sorted(split.values()).count('val'), 2)
        self.assertEqual(sorted(split.values()).count('test'), 2)

    def test_duplicate_bytes_keep_families_in_same_split(self):
        records = [{'family': f'light_{i:02}', 'sha256': str(i), 'classes': list(range(6))} for i in range(10)]
        records[1]['sha256'] = records[0]['sha256']
        split = split_families(records)
        self.assertEqual(split['light_00'], split['light_01'])

    def test_too_few_independent_groups_is_a_hard_error(self):
        records = [{'family': f'light_{i:02}', 'sha256': 'same', 'classes': list(range(6))} for i in range(5)]
        with self.assertRaises(ValueError):
            split_families(records)

    def test_preparation_preserves_raw_and_produces_matching_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            raw = base / 'raw'
            images, labels = raw / 'train/images', raw / 'train/labels'
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            originals = {}
            for i in range(10):
                stem = f'light_{i:02}_spur_01_1_600'
                path = images / (stem + '.png')
                Image.new('RGB', (64, 64), (i * 20, 30, 70)).save(path)
                label = labels / (stem.removesuffix('_600') + '_256.txt')
                label.write_text('\n'.join(f'{c} .5 .5 .2 .2' for c in range(6)))
                originals[path] = hashlib.sha256(path.read_bytes()).hexdigest()
                originals[label] = hashlib.sha256(label.read_bytes()).hexdigest()
            result = prepare_dataset(raw, base / 'prepared', base / 'reports')
            self.assertEqual(result['images'], 10)
            self.assertEqual(result['renamed_labels'], 10)
            self.assertTrue((base / 'prepared/dataset.yaml').exists())
            manifest = json.loads((base / 'prepared/manifest.json').read_text())
            self.assertEqual(len(manifest['records']), 10)
            for split in ('train', 'val', 'test'):
                outputs = list((base / 'prepared' / split / 'images').iterdir())
                self.assertTrue(outputs)
                for image in outputs:
                    self.assertTrue((base / 'prepared' / split / 'labels' / (image.stem + '.txt')).exists())
            for path, digest in originals.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == '__main__':
    unittest.main()
