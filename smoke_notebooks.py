"""Execute both notebooks on clearly synthetic data; no historical scores reproduced."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).parent.resolve()
with tempfile.TemporaryDirectory(prefix='ds-smoke-') as temp:
    data = Path(temp)
    rng = np.random.default_rng(42)
    for name, rows in [('train', 80), ('test', 20)]:
        frame = pd.DataFrame(rng.uniform(1, 5, (rows, 4)), columns=['longitude', 'latitude', 'rooms', 'income'])
        frame['median_house_value'] = frame.income * 10000 + frame.rooms * 1000
        frame.to_csv(data / f'california_housing_{name}.csv', index=False)
    descriptions = ['Fresh citrus apples white wine', 'Dark berries oak dry red wine',
                    'Light pear sweet summer wine', 'Ripe plum rich red wine']
    wine = pd.DataFrame({'description': descriptions * 40, 'price': [10, 30, 15, 25] * 40})
    wine.to_csv(data / 'winemag-data_first150k.csv')
    env = dict(os.environ, DATA_DIR=str(data), NOTEBOOK_SMOKE='1',
               NLTK_DATA=str(root / 'data' / 'nltk_data'))
    for name in ['google_california_pycaret.ipynb', 'NLP_wine_reviews.ipynb']:
        print(f'SYNTHETIC smoke: {name}', flush=True)
        subprocess.run([sys.executable, 'validate_notebooks.py', name], cwd=root, env=env, check=True)
