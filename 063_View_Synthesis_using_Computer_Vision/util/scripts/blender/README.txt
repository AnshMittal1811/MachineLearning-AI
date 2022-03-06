Usage info for blender 2.83.1 

1) Start blender

2) Load mesh: File (located in Header Bar) -> Import -> Stanford (.ply)

3) After changing static paths inside in main.py (input path should be the same as the one at step 2), import main.py script:
   3a) Click on "Scripting" (located in Header Bar)
   3b) Use Python console opened on the left
   3c) To import main.py:
       import sys 
       sys.path.append('/path/to/novel-view-synthesis/util/scripts/blender/')
       from main import *

4) See examples in main.py and call desired functions from the python console of blender. 

5) In the end make sure to call exporter function.