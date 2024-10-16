import sys, os.path, cv2, numpy as np

def otsu(img: np.ndarray) -> np.ndarray:
    hist, bin_edges = np.histogram(img, np.arange(257))
    N_t = hist.sum()
    mu_t = sum([i * hist[i] for i in range(256)]) / N_t
    omega_1 = 1 / N_t
    omega_2 = 1 - omega_1
    mu_1 = 1
    mu_2 = (mu_t - mu_1 * omega_1) / omega_2
    mu_diff = mu_1 - mu_2

    sigma_max = omega_1 * omega_2 * mu_diff * mu_diff
    threshold = 0

    for t in range(1, 256):
        omega_1 = sum([hist[i] for i in range(t)]) / N_t
        if omega_1 == 0:
            continue
        omega_2 = 1 - omega_1
        mu_1 = sum([i * hist[i] for i in range(t)]) / (N_t * omega_1)
        mu_2 = (mu_t - mu_1 * omega_1) / omega_2
        mu_diff = mu_1 - mu_2
        sigma = omega_1 * omega_2 * mu_diff * mu_diff
        if sigma > sigma_max:
            sigma_max = sigma
            threshold = t

    # print(threshold)
    result = img
    result[img >= threshold] = 255
    result[img < threshold] = 0

    return result


def main():
    assert len(sys.argv) == 3
    src_path, dst_path = sys.argv[1], sys.argv[2]

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = otsu(img)
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    main()
