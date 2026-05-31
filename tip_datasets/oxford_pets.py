"""tip_datasets/oxford_pets.py

Port of the ``OxfordPets.read_split`` / ``save_split`` static methods from
Tip-Adapter (https://github.com/gaopengcuhk/Tip-Adapter/blob/main/datasets/oxford_pets.py).

``Food101`` (and every other CoOp/Tip-Adapter dataset) reuses
``OxfordPets.read_split`` to parse a ``split_zhou_<Dataset>.json`` file into
train / val / test lists of :class:`~tip_datasets.utils.Datum` objects, with the
image directory prepended to each (relative) image path.

Only the static methods actually used by the Food101 split loader are kept here.
"""

import os

from .utils import Datum, read_json, write_json


class OxfordPets:
    """Container for the CoOp/Tip-Adapter split read/write helpers.

    The full upstream class also handles downloading and (re)building the
    Oxford-IIIT-Pets split; that machinery is unused on this branch (we only
    consume the pre-built ``split_zhou_Food101.json``), so only the split
    serialization helpers are ported.
    """

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(
                    impath=impath,
                    label=int(label),
                    classname=classname
                )
                out.append(item)
            return out

        print(f'Reading split from {filepath}')
        split = read_json(filepath)
        train = _convert(split['train'])
        val = _convert(split['val'])
        test = _convert(split['test'])

        return train, val, test

    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, '')
                if impath.startswith('/'):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {
            'train': train,
            'val': val,
            'test': test
        }

        write_json(split, filepath)
        print(f'Saved split to {filepath}')
