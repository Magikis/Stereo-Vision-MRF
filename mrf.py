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
    def __init__(self, smoothness_args, LABELS=16, init='ours'):
        self.size_of_labels = None
        self.LABELS = LABELS
        self.smoothness_cost = self._smoothness_cost(
            LAMBDA=1,
            SMOOTHNESS_TRUNC=8
        )

        border = LABELS + 2
        loaded_picture = np.load(
            'img_smaller_pbb_25_06-13:56.npy').astype(float)
        # print(loaded_picture)
        # _max = loaded_picture.max()
        # _max[ _max = np.inf] =
        # loaded_picture = (loaded_picture - loaded_picture.min()) / (_max - loaded_picture.min())
        # print(loaded_picture)
        self.mrf = np.zeros(
            (*loaded_picture.shape[:2], 5, LABELS),
        )
        self.mrf[:, :, Direction.data.value, :] = loaded_picture
        # self.mrf = self.mrf * (1 / self.mrf.sum(axis=-1)[..., np.newaxis])

        self.smoothness_cost = self._smoothness_cost(
            **smoothness_args
        )

        self.mrf = self.mrf[
            border: self.mrf.shape[0] - border,
            border: self.mrf.shape[1] - border
        ]

        self.mrf[:, :, Direction.data.value, :] = (
            (
                self.mrf[:, :, Direction.data.value, :]
                - self.mrf[:, :, Direction.data.value, :].min(axis=-1)[...,np.newaxis]
            )
            / (
                self.mrf[:, :, Direction.data.value, :].max(axis=-1)
                - self.mrf[:, :, Direction.data.value, :].min(axis=-1)
            )[...,np.newaxis]
        )
        self.map()

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
