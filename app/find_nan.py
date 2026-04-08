import app.environment as env
import math
e = env.DataQualityEnvironment()
e.reset('full_cleanup')
d = e.state()['observation']

def find_nan(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            find_nan(v, path + '.' + k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_nan(v, path + f'[{i}]')
    elif isinstance(obj, float) and math.isnan(obj):
        print('Found NaN at:', path)

find_nan(d)
