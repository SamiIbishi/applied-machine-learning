import os
import sys
os.system("nohup sh -c '" +
          sys.executable + " hyperparameter_tuning.py >res1.txt" + "' &")