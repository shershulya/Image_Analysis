import sys, os.path, cv2, numpy as np


def box_filter(img: np.ndarray, w: int, h: int) -> np.ndarray:
    # print(img)

    integr_sum = img.astype(np.uint64)    
    img_h = img.shape[0]
    img_w = img.shape[1]

    for j in range(1, img_w):
        integr_sum[0][j] += integr_sum[0][j - 1]

    for i in range(1, img_h):
        integr_sum[i][0] += integr_sum[i - 1][0]

    for i in range(1, img_h):
        for j in range(1, img_w):
            integr_sum[i][j] = img[i][j] + integr_sum[i][j - 1] + integr_sum[i - 1][j] - integr_sum[i - 1][j - 1]

    result = img
    h_half = h // 2
    w_half = w // 2
    # for i in range(h_half, img_h - h_half):
    #     for j in range(w_half, img_w - w_half):
    #         result[i][j] = integr_sum[i + h_half][j + w_half] \
    #                      + integr_sum[i - h_half - 1][j - w_half - 1] \
    #                      - integr_sum[i + h_half][j - w_half - 1] \
    #                      - integr_sum[i - h_half - 1][j + w_half] \

    integr_sum = np.pad(integr_sum, ((h - h_half, h - h_half), (w - w_half, w - w_half)), 'edge')
    # print(integr_sum)

    for i in range(img_h):
        for j in range(img_w):
            result[i][j] = np.round((integr_sum[i][j] \
                         - integr_sum[i][j + w] \
                         - integr_sum[i + h][j] \
                         + integr_sum[i + h][j + w]) \
                         / (w * h))

    return result

def main():
    assert len(sys.argv) == 5
    src_path, dst_path = sys.argv[1], sys.argv[2]
    w, h = int(sys.argv[3]), int(sys.argv[4])
    assert w > 0
    assert h > 0

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    # result = cv2.blur(img, (w, h))
    result = box_filter(img, w, h)
    print(result)
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    main()
