import random
import os

import cv2
import numpy as np

from range_compression import RangeCompressedMask, mask_encode


def test_mask():
    width, height = 800, 600
    image = np.zeros((height, width), dtype=np.uint8)

    num_polygons = 5 

    for _ in range(num_polygons):
        num_sides = random.randint(3, 8)
        color = random.randint(0, 255)
        vertices = []

        for _ in range(num_sides):
            x = random.randint(0, width)
            y = random.randint(0, height)
            vertices.append((x, y))

        pts = np.array(vertices, np.int32)

        cv2.fillPoly(image, [pts], color)

    rcm = mask_encode(image)
    assert isinstance(rcm, RangeCompressedMask)
    assert rcm.w == width and rcm.h == height
    assert len(rcm.row_indexes) == height
    
    random_n = 10000
    randx, randy = np.random.randint(0, width, random_n), np.random.randint(0, height, random_n)
    
    res = rcm.find_index(randx, randy)
    actual_res = image[randy, randx]
    
    try:
        assert np.array_equal(res, actual_res)
    except:
        np.save('tests/image.npy', image)
        np.save('tests/randx.npy', randx)
        np.save('tests/randy.npy', randy)
        np.save('tests/res.npy', res)
        np.save('tests/actual_res.npy', actual_res)
        raise

    rcm.save('tests/output')
    rcm2 = RangeCompressedMask.load('tests/output')
    assert np.array_equal(rcm.encodings[:, :3], rcm2.encodings)
    
    actual_res2 = rcm2.find_index(randx, randy)
    assert np.array_equal(actual_res, actual_res2)

    os.remove('tests/output/encodings.parquet')
    os.remove('tests/output/row_indexes.parquet')
    os.remove('tests/output/meta.json')
    os.rmdir('tests/output')

def test_benchmark(benchmark):
    benchmark(test_mask)
