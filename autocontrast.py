import sys, os.path, cv2, numpy as np

def autocontrast(img: np.ndarray, white_percent: float, black_percent: float) -> np.ndarray:
    n_bins = 256
    n = img.size
    cut_white = white_percent * n
    cut_black = black_percent * n
    if cut_white == 0 and cut_black == 0:
        high, low = img.max(), img.min()
    else:
        hist = cv2.calcHist([img], [0], None, [n_bins], [0, n_bins])
        low = np.argwhere(np.cumsum(hist) > cut_black)
        low = 0 if low.shape[0] == 0 else low[0]
        high = np.argwhere(np.cumsum(hist[::-1]) > cut_white)
        high = n_bins - 1 if high.shape[0] == 0 else n_bins - 1 - high[0]
    if high <= low:
        table = np.arange(n_bins)
    else:
        scale = (n_bins - 1) / (high - low)
        offset = -low * scale
        table = np.arange(n_bins) * scale + offset
        table[table < 0] = 0
        table[table > n_bins - 1] = n_bins - 1
    table = table.clip(0, 255).astype(np.uint8)
    return table[img]


def main():
    assert len(sys.argv) == 5
    src_path, dst_path = sys.argv[1], sys.argv[2]
    white_percent, black_percent = float(sys.argv[3]), float(sys.argv[4])
    assert 0 <= white_percent < 1
    assert 0 <= black_percent < 1

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = autocontrast(img, white_percent, black_percent)
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    main()
