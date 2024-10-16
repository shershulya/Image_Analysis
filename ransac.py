import sys, os.path, json, numpy as np

import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2

def generate_data(
        img_size: tuple, line_params: tuple,
        n_points: int, sigma: float, inlier_ratio: float
) -> np.ndarray:
    line_points = int(n_points * inlier_ratio)
    outlier_points = n_points - line_points

    line_x = np.random.choice(range(img_size[0]), size=line_points, replace=False)
    line_y = (- (line_params[0] * line_x + line_params[2]) / line_params[1]).astype(np.int64)

    for i in range(line_points):
        noise_x = np.random.normal(0, sigma, 1)
        line_x[i] = line_x[i] + noise_x
        if (line_x[i] >= img_size[0]):
            line_x[i] = img_size[0] - 1
        if (line_x[i] < 0):
            line_x[i] = 0
        noise_y = np.random.normal(0, sigma, 1)
        line_y[i] = line_y[i] + noise_y
        if (line_y[i] >= img_size[1]):
            line_y[i] = img_size[1] - 1
        if (line_y[i] < 0):
            line_y[i] = 0

    outlier_x = np.random.choice(range(img_size[0]), size=outlier_points, replace=False)
    outlier_y = np.random.choice(range(img_size[1]), size=outlier_points, replace=False)

    x = np.concatenate((line_x, outlier_x))
    y = np.concatenate((line_y, outlier_y))

    plt.scatter(x, y, s=3)
    plt.xlim(0, img_size[0]), plt.ylim(0, img_size[1])

    data = np.c_[x, y]
    return data


def compute_ransac_threshold(
        alpha: float, sigma: float
) -> float:
    threshold_sq = chi2.ppf(alpha, df=2) * sigma * sigma
    return threshold_sq


def compute_ransac_iter_count(
        conv_prob: float, inlier_ratio: float
) -> int:
    iters = int(np.round(np.log(1 - conv_prob) / np.log(1 - inlier_ratio ** 2)))
    return iters


def compute_line_ransac(
        data: np.ndarray, threshold: float, iter_count: int
) -> tuple:
    best_score = 0
    detected_line = ()
    detected_points = ()

    for _ in range(iter_count):
        ##### Since the lecture did not tell us about the choice of parameter m - the set for comparison, #####################
        ##### I chose the minimum set of 2 points to draw a line ##############################################################
        sample_idx = np.random.choice(len(data), size=2, replace=False)

        x1 = data[sample_idx[0]][0]
        x2 = data[sample_idx[1]][0]
        y1 = data[sample_idx[0]][1]
        y2 = data[sample_idx[1]][1]

        a = y1 - y2
        b = x2 - x1
        c = y1 * (x1 - x2) + x1 * (y2 - y1)

        score = 0
        for i in range(data.shape[0]):
            m_x = data[i][0]
            m_y = data[i][1]
            tmp = (a * m_x + b * m_y + c)
            numerator = tmp * tmp
            denominator = a * a + b * b
            distance = numerator / denominator
            if distance <= threshold:
                score += 1

        if score > best_score:
            best_score = score
            detected_line = (a, b, c)
            detected_points = (x1, y1, x2, y2)

    plt.plot([detected_points[0], detected_points[2]], [detected_points[1], detected_points[3]], marker = 'o')
    # plt.show()
    ################ If You want to draw the generated points and the result line just uncomment the line above ###############
    return detected_line


def detect_line(params: dict) -> tuple:
    data = generate_data(
        (params['w'], params['h']),
        (params['a'], params['b'], params['c']),
        params['n_points'], params['sigma'], params['inlier_ratio']
    )
    threshold = compute_ransac_threshold(
        params['alpha'], params['sigma']
    )
    iter_count = compute_ransac_iter_count(
        params['conv_prob'], params['inlier_ratio']
    )
    detected_line = compute_line_ransac(data, threshold, iter_count)
    return detected_line


def main():
    assert len(sys.argv) == 2
    params_path = sys.argv[1]
    assert os.path.exists(params_path)
    with open(params_path) as fin:
        params = json.load(fin)
    assert params is not None

    """
    params:
    line_params: (a,b,c) - line params (ax+by+c=0)
    img_size: (w, h) - size of the image
    n_points: count of points to be used

    sigma - Gaussian noise
    alpha - probability of point is an inlier

    inlier_ratio - ratio of inliers in the data
    conv_prob - probability of convergence
    """

    detected_line = detect_line(params)
    print(detected_line)


if __name__ == '__main__':
    main()
