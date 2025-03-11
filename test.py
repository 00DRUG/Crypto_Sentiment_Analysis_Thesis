import pandas as pd

def _get_spans(sparse_labels):
    # calculate column/row spans
    spans = [i for i, label in enumerate(sparse_labels) if label]
    spans.append(len(sparse_labels))
    # difference between consecutive elements
    spans = [x - spans[i - 1] for i, x in enumerate(spans) if i > 0]
    sparse_spans = []
    popped = False
    for i in sparse_labels:
        if i:
            sparse_spans.append(spans.pop(0))
            popped = True
        elif popped:
            sparse_spans.append(0)
    return sparse_spans

def get_sparse_labels(multiindex, transpose=True):
    sparse_labels = multiindex.format(sparsify=True, adjoin=False)


    if not isinstance(sparse_labels[0], tuple):
        sparse_labels = [tuple(sparse_labels)]
    sparse_spans = [_get_spans(labels) for labels in sparse_labels]
    # transpose the lists of tuples
    if transpose is True:
        sparse_labels = list(zip(*sparse_labels))
        sparse_spans = list(zip(*sparse_spans))
    # zip into (label, span) pairs
    zipped = []
    for a, b in zip(sparse_labels, sparse_spans):
        r = []
        for pair in zip(a, b):
            r.append(pair)
        zipped.append(r)
    return zipped
# Sample MultiIndex
idx = pd.MultiIndex.from_tuples([
    ('BC', '15', '0.0942809042', '960'),
    ('BC', '30', '0.0471404521', '3720'),
    ('BC', '60', '0.023570226', '14640'),
    ('BC', '120', '0.011785113', '58080'),
    ('BC', '240', '0.0058925565', '231360'),
    ('BC', '15', '0.0942809042', '960'),
    ('BC', '30', '0.0471404521', '3720'),
    ('BC', '60', '0.023570226', '14640'),
    ('VG', '120', '0.011785113', '58080'),
    ('VG', '240', '0.0058925565', '231360')
], names=[None, 'N_x', 'h', 'DOF'])

# Test the function
sparse_labels = [
    [value if (i == 0 or value != level[i-1]) else '' for i, value in enumerate(level)]
    for level in zip(*idx.tolist())
]



# Display the result
for row in sparse_labels:
    print(row)
