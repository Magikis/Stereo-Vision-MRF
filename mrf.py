import cv2
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt


class Direction(Enum):
    left = 0
    right = 1
    up = 2
    down = 3
    data = 4


class MRFStereo():
    def __init__(self, left_image, right_image, LABELS=16, init='ours'):
        self.left_image = left_image
        self.right_image = right_image
        self.mrf = np.zeros(
            (*left_image.shape, 5, LABELS), dtype="int64"
        )
        self.size_of_labels = None
        self.LABELS = LABELS
        self.smoothness_cost = self._smoothness_cost(
            LAMBDA=1,
            SMOOTHNESS_TRUNC=8
        )

        border = LABELS + 2
        if init == 'ours':
            for y in range(border, self.mrf.shape[0] - border):
                for x in range(border, self.mrf.shape[1] - border):
                    for l in range(LABELS):
                        self.mrf[y, x, Direction.data.value,
                                 l] = self.data_cost_stereo(x, y, l)

        if init == 'cv':
            stereo = cv2.StereoBM_create(16, 5)
            disparity = stereo.compute(left_image, right_image)
            disparity = (((disparity + 16) / (disparity + 16).max())
                         * 15).astype(np.int)
            for y in range(border, self.mrf.shape[0] - border):
                for x in range(border, self.mrf.shape[1] - border):
                    # for l in range(LABELS):
                    self.mrf[y, x, Direction.data.value] = \
                        self.smoothness_cost[disparity[y, x], :].copy(
                    )
        self.smoothness_cost = self._smoothness_cost(
            LAMBDA=100,
            SMOOTHNESS_TRUNC=3,
            squared=True
        )
        
        self.mrf = self.mrf[
            border: self.mrf.shape[0] - border,
            border: self.mrf.shape[1] - border
        ]
        self.map()

    def data_cost_stereo(self, x, y, label):
        r = 2
        low_y = y - r
        high_y = y + r + 1
        low_x = x - r
        high_x = x + r + 1
        assert(low_x - label >= 0)
        # if low_x - label < 0:
        #     return 0

        left = self.left_image[low_y: high_y, low_x:high_x]
        right = self.right_image[low_y:high_y, low_x - label: high_x - label]

        if left.shape != right.shape:
            print(y, x, left.shape, right.shape)
            assert(0)
        return np.abs(left - right).mean()

    def _smoothness_cost(self, LAMBDA=20, SMOOTHNESS_TRUNC=2, squared=False):
        i = np.array(
            list(np.arange(self.LABELS)) * self.LABELS).reshape(self.LABELS, self.LABELS)
        j = i.copy().T
        temp = np.abs(i-j) ** (2 if squared else 1)
        temp[temp > SMOOTHNESS_TRUNC] = SMOOTHNESS_TRUNC
        return LAMBDA * temp

    def believe_propagation(self, direction):
        if direction == Direction.right:
            indexs = [
                (y, x)
                for y in range(0, self.mrf.shape[0])
                for x in range(0, self.mrf.shape[1] - 1)
            ]
        elif direction == Direction.left:
            indexs = [
                (y, x)
                for y in range(0, self.mrf.shape[0])
                for x in range(self.mrf.shape[1] - 1, 0, -1)
            ]
        elif direction == Direction.down:
            indexs = [
                (y, x)
                for x in range(0, self.mrf.shape[1])
                for y in range(0, self.mrf.shape[0] - 1)
            ]
        elif direction == Direction.up:
            indexs = [
                (y, x)
                for x in range(0, self.mrf.shape[1])
                for y in range(self.mrf.shape[0] - 1, 0, -1)
            ]
        for index in indexs:
            self.send_msg(*index, direction)

    def send_msg(self, y, x, direction):

        new_msg = np.zeros(self.LABELS)
        width = self.mrf.shape[1]

        p = self.smoothness_cost.copy()
        p += np.tile(
            np.sum(self.mrf[y, x, :, :], axis=-2) -
            self.mrf[y, x, direction.value, :], (self.LABELS, 1)
        )
        new_msg = np.min(p, axis=1)

        if direction == Direction.left:
            self.mrf[y, x-1, direction.value, :] = new_msg[:]
        elif direction == Direction.right:
            self.mrf[y, x+1, direction.value, :] = new_msg[:]
        elif direction == Direction.up:
            self.mrf[(y-1), x, direction.value, :] = new_msg[:]
        elif direction == Direction.down:
            self.mrf[(y+1), x, direction.value, :] = new_msg[:]
        else:
            assert(0)

    def map(self):
        costs = self.mrf.sum(axis=2)
        self.best_assignment = costs.argmin(axis=-1)

        energy = 0
        data = [
            (self.best_assignment[1:, :], self.best_assignment[:-1, :]),
            (self.best_assignment[:-1, :], self.best_assignment[1:, :]),
            (self.best_assignment[:, 1:], self.best_assignment[:, :-1]),
            (self.best_assignment[:, :-1], self.best_assignment[:, 1:]),
        ]
        for x in data:
            ids = np.stack(x, axis=-1)
            energy += self.smoothness_cost[ids[..., 0], ids[..., 1]].sum()
        energy += self.best_assignment[..., Direction.data.value].sum()

        return energy

    def run(self, BP_ITERATIONS=25, verbose=False):

        for i in range(BP_ITERATIONS):
            self.believe_propagation(Direction.right)
            self.believe_propagation(Direction.left)
            self.believe_propagation(Direction.up)
            self.believe_propagation(Direction.down)
            energy = self.map()
            print(f"iteration {(i+1)} / {BP_ITERATIONS}, energy = {energy}")

            if verbose:
                self.show()
        return self.show()

    def show(self, image=None, modify=False):
        if image is None:
            image = self.best_assignment
        output = (image * (256/self.LABELS)).astype("int64")
        plt.imshow(output, cmap='gray', vmin=0, vmax=255)
        plt.show()
        return output


def read_data(debug=False):
    left = "tsukuba-imL.png"
    right = "tsukuba-imR.png"

    img_left = cv2.imread(left, 0)
    img_right = cv2.imread(right, 0)
    if debug:
        img_left = img_left[:50, :80]
        img_right = img_right[:50, :80]
    return img_left, img_right


# if __name__ == "__main__":
#     mrf = MRFStereo(*read_data(), init=False)
#     mrf.mrf = np.zeros((3, 3, 5, 6))
#     x = mrf.believe_propagation('DOWN')
