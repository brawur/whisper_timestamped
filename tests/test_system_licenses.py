import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "system_licenses", Path(__file__).parents[1] / "scripts" / "system_licenses.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SystemLicenseTests(unittest.TestCase):
    def test_notices_sources_and_shared_texts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docs, common = root / "doc", root / "common"
            (docs / "gcc").mkdir(parents=True)
            (docs / "gcc-dev").symlink_to(docs / "gcc", target_is_directory=True)
            common.mkdir()
            notice = "Copyright: example\nLicense: GPL-3+\nSee /usr/share/common-licenses/GPL-3\n"
            (docs / "gcc" / "copyright").write_text(notice)
            (common / "GPL-3").write_text("Complete license text")
            entries, sources = module.collect(
                "gcc\t14\tgcc-14\t14-1\tinstalled\n"
                "gcc-dev\t14\tgcc-14\t14-1\tinstalled\n"
                "removed\t1\tremoved\t1\tconfig-files\n", docs, common
            )
            self.assertEqual(len(entries), 3)
            self.assertEqual(entries[0]["LicenseText"], notice)
            self.assertEqual(entries[0]["License"], "GPL-3+")
            self.assertEqual(entries[1]["LicenseText"], notice)
            self.assertEqual(entries[2]["LicenseText"], "Complete license text")
            self.assertEqual(sources, [{"package": "gcc-14", "version": "14-1"}])

    def test_missing_notice_fails_build(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                module.collect("gcc\t14\tgcc-14\t14-1\tinstalled\n",
                               Path(directory), Path(directory))


if __name__ == "__main__":
    unittest.main()
