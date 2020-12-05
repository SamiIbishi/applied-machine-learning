import unittest
import os
import pep8



class MyTestCase(unittest.TestCase):
    def getFilePaths(self, rootdir: str = "..\src"):
        """
        determines recursively all python (*.py) files of the given rootdir

        :param rootdir: the dir where to start
        :return: an array listing all filepaths
        """

        paths = []
        dirpaths = os.listdir(rootdir)
        for path in dirpaths:
            path = os.path.join(rootdir, path)
            if os.path.isdir(path):
                paths = paths + self.getFilePaths(path)
            elif path[-3:] == ".py":
                paths = paths + [path]

        return paths

    def test_pep8_conformance(self):
        """
        Test that we conform to PEP8.

        :return:
        """

        # setup
        paths = self.getFilePaths()
        pep8style = pep8.StyleGuide()
        pep8style.options.max_line_length = 100

        print("paths: ", paths)

        # execute
        result = pep8style.check_files(paths)

        # validate
        self.assertEqual(result.total_errors, 0, "Found code style errors (and warnings).")

if __name__ == '__main__':
    unittest.main()
