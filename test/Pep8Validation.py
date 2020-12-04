import unittest
import os
import pep8



class MyTestCase(unittest.TestCase):
    def getFilePaths(self, rootdir="..\src"):
        paths = []
        dirpaths = os.listdir(rootdir)
        for path in dirpaths:
            path = os.path.join(rootdir, path)
            if os.path.isdir(path):
                paths = paths + self.getFilePaths(path)
            else:
                paths = paths + [path]

        return paths

    def test_pep8_conformance(self):
        """Test that we conform to PEP8."""

        paths = self.getFilePaths()
        pep8style = pep8.StyleGuide()

        result = pep8style.check_files(paths)

        self.assertEqual(result.total_errors, 0, "Found code style errors (and warnings).")


if __name__ == '__main__':
    unittest.main()
