import random
import os

import cv2
import numpy as np

from range_compression import RangeCompressedMask, mask_encode

def get_rand_image():
    width, height = 1024, 2048
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
    return image

def _test_encode(image):
    height, width = image.shape

    rcm = mask_encode(image)
    assert isinstance(rcm, RangeCompressedMask)
    assert rcm.w == width and rcm.h == height
    assert len(rcm.row_indexes) == height
    return rcm

def get_randxy(width, height):
    random_n = 10000
    randx, randy = np.random.randint(0, width, random_n), np.random.randint(0, height, random_n)
    return randx, randy

def _test_find_mask(rcm, image, randx, randy):
    res = rcm.find_index(randx, randy)
    actual_res = image[randy, randx]
    assert np.array_equal(res, actual_res)
    return actual_res


def test_mask(benchmark):
    image = get_rand_image()
    height, width = image.shape

    rcm = _test_encode(image)
    randx, randy = get_randxy(width, height)

    benchmark.pedantic(_test_find_mask, args=(rcm, image, randx, randy), iterations=10, rounds=50)
    
    actual_res = _test_find_mask(rcm, image, randx, randy)
    rcm.save('tests/output')
    rcm2 = RangeCompressedMask.load('tests/output')
    assert np.array_equal(rcm.encodings[:, :3], rcm2.encodings)
    
    actual_res2 = rcm2.find_index(randx, randy)
    assert np.array_equal(actual_res, actual_res2)

    os.remove('tests/output/encodings.parquet')
    os.remove('tests/output/row_indexes.parquet')
    os.remove('tests/output/meta.json')
    os.rmdir('tests/output')

