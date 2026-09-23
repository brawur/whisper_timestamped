import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('source_inventory', Path(__file__).parents[1] / 'scripts/source_inventory.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SourceInventoryTests(unittest.TestCase):
    def test_debian_source_version_deduplicated_and_all_files_preserved(self):
        calls = []
        def fetch(url):
            calls.append(url)
            return {'package': 'gcc-14', 'version': '1:14-1', 'result': [{'hash': 'a' * 40}, {'hash': 'b' * 40}]}
        report = [dict(Name=name, Version='14-1+b2', License='GPL', PackageSystem='deb',
                       SourcePackage='gcc-14', SourceVersion='1:14-1') for name in ('debian/gcc', 'debian/libgcc')]
        entries = module.inventory(report, fetch)
        self.assertEqual(len(calls), 1)
        self.assertIn('1%3A14-1', calls[0])
        self.assertEqual(len(entries[0]['artifacts']), 2)
        self.assertEqual(entries[1]['status'], 'metadata-resolved')
        self.assertEqual(entries[0]['version'], '14-1+b2')

    def test_version_normalization_does_not_hide_local_or_prerelease_versions(self):
        self.assertTrue(module.same_version('13.0.3', '13.0.3.0'))
        self.assertFalse(module.same_version('2.0', '2.0+cu130'))
        self.assertFalse(module.same_version('2.0', '2.0rc1'))

    def test_only_source_archives_not_wheels(self):
        entry = dict(ecosystem='pypi', version='1', metadata_url='test')
        result = module.resolve(entry, lambda _: {'info': {'version': '1'}, 'urls': [
            {'packagetype': 'bdist_wheel'},
            {'packagetype': 'sdist', 'url': 'https://example.org/source.tar.gz', 'filename': 'source.tar.gz', 'digests': {'sha256': 'abc'}}]})
        self.assertEqual(len(result['artifacts']), 1)
        self.assertEqual(result['artifacts'][0]['sha256'], 'abc')

    def test_missing_sources_wrong_version_and_network_failure_stay_open(self):
        entry = dict(ecosystem='pypi', version='1', metadata_url='test')
        for info in ({'info': {'version': '1'}, 'urls': []}, {'info': {'version': '2'}, 'urls': []}):
            self.assertEqual(module.resolve(entry, lambda _: info)['status'], 'review-required')
        def fail(_):
            raise TimeoutError('unavailable')
        self.assertEqual(module.resolve(entry, fail)['reason'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
