import json
from psycopg2.extras import RealDictCursor
#import cv2
import psycopg2
import cv2


CONNECTION = "postgres://postgres:"

conn = psycopg2.connect(CONNECTION)
cursor = conn.cursor(cursor_factory=RealDictCursor)


def get_sample():
    camera_name, camera_id = 'cam2', 4
    
    print('Executing SQL command')

    cursor.execute("SELECT * FROM annotations WHERE camera_id = {} and time >='2021-05-01 00:00:00' and time <='2021-05-07 23:59:50' and label_id in (1,2)".format(camera_id))

    print('Dumping to json')
    annotations = json.dumps(cursor.fetchall(), indent=2, default=str)
    wjdata = json.loads(annotations)
    with open('{}_{}_test.json'.format(camera_name, camera_id), 'w') as f:
        json.dump(wjdata, f)
    print('Done dumping to json')
    
get_sample()
