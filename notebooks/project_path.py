import sys
import os


# Allows to import src files,
# if jupyter notebook doesn't find them directly.
# Source:
# - Comment 1: https://stackoverflow.com/a/35273613
# - Comment 2: https://stackoverflow.com/a/51028921
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
