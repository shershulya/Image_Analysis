import sys, os.path, cv2, numpy as np

from tqdm import tqdm

def gradient_img(img: np.ndarray) -> np.ndarray:
    hor_grad = (img[1:, :] - img[:-1, :])[:, :-1]
    ver_grad = (img[:, 1:] - img[:, :-1])[:-1:, :]
    magnitude = np.sqrt(hor_grad ** 2 + ver_grad ** 2)
    return magnitude


def hough_transform(
        img: np.ndarray, theta_step: float, rho_step: float
) -> (np.ndarray, list, list):
    
    img = 255 * (img / img.max())
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > 127:
                img[i][j] = 255
            else:
                img[i][j] = 0

    height, width = img.shape
    rho_max = int(np.sqrt(height * height + width * width))
    thetas = list(np.arange(0, np.pi, theta_step))
    rhos = list(np.arange(-rho_max, rho_max + 1, rho_step))
    accumulator = np.zeros((len(rhos), len(thetas)))

    for i in tqdm(range(width)):
        for j in range(height):
            if img[j][i]:
                for theta_idx, theta in enumerate(thetas):
                    rho = j * np.sin(theta) + i * np.cos(theta)
                    rho_idx = np.argmin(np.abs(rhos - rho))
                    accumulator[rho_idx, theta_idx] += 1
    return accumulator, thetas, rhos

def get_lines(
        ht_map: np.ndarray, n_lines: int,
        thetas: list, rhos: list,
        min_delta_rho: float, min_delta_theta: float
) -> list:
    coeff = []
    top_r, top_t = np.unravel_index(np.argsort(-ht_map.flatten()), ht_map.shape)
    lines = 0

    while lines != n_lines:
        rho = rhos[top_r[0]]
        theta = thetas[top_t[0]]

        rhos_used = np.array(rho)
        thetas_used = np.array(theta)

        k = -1 * np.cos(theta) / np.sin(theta)
        b = rho / np.sin(theta)
        coeff.append((k, b))

        lines += 1
        if lines == n_lines:
            break

        for r, t in zip(top_r[1:], top_t[1:]):
            rho = rhos[r]
            theta = thetas[t]

            if (min_delta_rho > np.min(np.abs(rhos_used - rho))):
                continue
            if (min_delta_theta > np.min(np.abs(thetas_used - theta))):
                continue
            
            k = -1 * np.cos(theta) / np.sin(theta)
            b = rho / np.sin(theta)
            coeff.append((k, b))
            
            lines += 1
            if lines == n_lines:
                break
    return coeff


def main():
    assert len(sys.argv) == 9
    src_path, dst_ht_path, dst_lines_path, theta, rho, \
        n_lines, min_delta_rho, min_delta_theta = sys.argv[1:]

    theta = float(theta)
    assert theta > 0.0

    rho = float(rho)
    assert rho > 0.0

    n_lines = int(n_lines)
    assert n_lines > 0

    min_delta_rho = float(min_delta_rho)
    assert min_delta_rho > 0.0

    min_delta_theta = float(min_delta_theta)
    assert min_delta_theta > 0.0

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    gradient = gradient_img(img.astype(float))

    ht_map, thetas, rhos = hough_transform(gradient, theta, rho)
    cv2.imwrite(dst_ht_path, ht_map)

    lines = get_lines(
        ht_map, n_lines, thetas, rhos, min_delta_rho, min_delta_theta
    )

    with open(dst_lines_path, 'w') as fout:
        for line in lines:
            fout.write(f'{line[0]:.3f}, {line[1]:.3f}\n')


if __name__ == '__main__':
    main()