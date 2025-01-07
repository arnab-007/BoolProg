import subprocess
import random
import re


result = subprocess.run(['./cmsgen/build/cmsgen', str('input-cnf')], 
                            capture_output=True, text=True)
output = result.stdout.strip()
print(output)