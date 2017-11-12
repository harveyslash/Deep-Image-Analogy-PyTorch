"""
The Patchmatch Algorithm. The actual algorithm is a nearly
line to line port of the original c++ version.
The distance calculation is different to leverage numpy's vectorized
operations.

This version uses 4 images instead of 2.
You can supply the same image twice to use patchmatch between 2 images.

"""

import numpy as np
import matplotlib.pyplot as plt


class PatchMatch(object):
    def __init__(self, a, aa, b, bb, patch_size):
        """
        Initialize Patchmatch Object.
        This method also randomizes the nnf , which will eventually
        be optimized.
        """
        assert a.shape == b.shape == aa.shape == bb.shape, "Dimensions were unequal for patch-matching input"
        self.A = a
        self.B = b
        self.AA = aa
        self.BB = bb
        self.patch_size = patch_size
        self.nnf = np.zeros(shape=(2, self.A.shape[0], self.A.shape[1])).astype(np.int)  # the nearest neighbour field
        self.nnd = np.zeros(shape=(self.A.shape[0], self.A.shape[1]))  # the distance map for the nnf
        self.initialise_nnf()

    def initialise_nnf(self):
        """
        Set up a random NNF
        Then calculate the distances to fill up the NND
        :return:
        """
        self.nnf[0] = np.random.randint(self.B.shape[1], size=(self.A.shape[0], self.A.shape[1]))
        self.nnf[1] = np.random.randint(self.B.shape[0], size=(self.A.shape[0], self.A.shape[1]))
        self.nnf = self.nnf.transpose((1, 2, 0))
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                pos = self.nnf[i, j]
                self.nnd[i, j] = self.cal_dist(i, j, pos[1], pos[0])

    def cal_dist(self, ay, ax, by, bx):
        """
        Calculate distance between a patch in A to a patch in B.
        :return: Distance calculated between the two patches
        """
        dx0 = dy0 = self.patch_size // 2
        dx1 = dy1 = self.patch_size // 2 + 1
        dx0 = min(ax, bx, dx0)
        dx1 = min(self.A.shape[0] - ax, self.B.shape[0] - bx, dx1)
        dy0 = min(ay, by, dy0)
        dy1 = min(self.A.shape[1] - ay, self.B.shape[1] - by, dy1)
        return np.sum(((self.A[ay - dy0:ay + dy1, ax - dx0:ax + dx1] - self.B[by - dy0:by + dy1, bx - dx0:bx + dx1]) ** 2) + (
            (self.AA[ay - dy0:ay + dy1, ax - dx0:ax + dx1] - self.BB[by - dy0:by + dy1, bx - dx0:bx + dx1]) ** 2)) / ((dx1 + dx0) * (dy1 + dy0))

    def reconstruct(self):
        """
        Simple reconstruction of A using patches from B.
        :return: The reconstructed RGB Matrix
        """
        ans = np.zeros_like(self.A)
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                pos = self.nnf[i, j]
                ans[i, j] = self.B[pos[1], pos[0]]
        return ans

    def reconstruct_avg(self, patch_size=5):

        final = np.zeros_like(self.B)
        for i in range(self.B.shape[0]):
            for j in range(self.B.shape[1]):

                dx0 = dy0 = patch_size // 2
                dx1 = dy1 = patch_size // 2 + 1
                dx0 = min(j, dx0)
                dx1 = min(self.B.shape[0] - j, dx1)
                dy0 = min(i, dy0)
                dy1 = min(self.B.shape[1] - i, dy1)

                patch = self.nnf[i - dy0:i + dy1, j - dx0:j + dx1]

                lookups = np.zeros(shape=(patch.shape[0], patch.shape[1], 3), dtype=np.float32)

                for ay in range(patch.shape[0]):
                    for ax in range(patch.shape[1]):
                        x, y = patch[ay, ax]
                        lookups[ay, ax] = self.B[y, x]

                if lookups.size > 0:
                    value = np.average(lookups, axis=(0, 1))
                    final[i, j] = value

        return final

    def visualize(self):
        """
        Get the NNF visualisation
        :return: The RGB Matrix of the NNF
        """
        nnf = self.nnf

        img = np.zeros((nnf.shape[0], nnf.shape[1], 3), dtype=np.uint8)

        for i in range(nnf.shape[0]):
            for j in range(nnf.shape[1]):
                pos = nnf[i, j]
                img[i, j, 0] = int(255 * (pos[0] / self.B.shape[1]))
                img[i, j, 2] = int(255 * (pos[1] / self.B.shape[0]))

        return img

    def propagate(self, iters=2, rand_search_radius=200):
        """
        Optimize the NNF using PatchMatch Algorithm
        :param iters: number of iterations
        :param rand_search_radius: max radius to use in random search
        :return:
        """
        a_cols = self.A.shape[1]
        a_rows = self.A.shape[0]

        b_cols = self.B.shape[1]
        b_rows = self.B.shape[0]

        for it in range(iters):
            ystart = 0
            yend = a_rows
            ychange = 1
            xstart = 0
            xend = a_cols
            xchange = 1

            if it % 2 == 1:
                xstart = xend - 1
                xend = -1
                xchange = -1
                ystart = yend - 1
                yend = -1
                ychange = -1

            ay = ystart
            while ay != yend:

                ax = xstart
                while ax != xend:

                    xbest, ybest = self.nnf[ay, ax]
                    dbest = self.nnd[ay, ax]
                    if ax - xchange < a_cols and ax - xchange >= 0:
                        vp = self.nnf[ay, ax - xchange]
                        xp = vp[0] + xchange
                        yp = vp[1]
                        if xp < b_cols and xp >= 0:
                            val = self.cal_dist(ay, ax, yp, xp)
                            if val < dbest:
                                xbest, ybest, dbest = xp, yp, val

                    if abs(ay - ychange) < a_rows and ay - ychange >= 0:
                        vp = self.nnf[ay - ychange, ax]
                        xp = vp[0]
                        yp = vp[1] + ychange
                        if yp < b_rows and yp >= 0:
                            val = self.cal_dist(ay, ax, yp, xp)
                            if val < dbest:
                                xbest, ybest, dbest = xp, yp, val
                    if rand_search_radius is None:
                        rand_d = max(self.B.shape[0], self.B.shape[1])
                    else:
                        rand_d = rand_search_radius

                    while rand_d >= 1:
                        try:
                            xmin = max(xbest - rand_d, 0)
                            xmax = min(xbest + rand_d, b_cols)

                            ymin = max(ybest - rand_d, 0)
                            ymax = min(ybest + rand_d, b_rows)

                            if xmin > xmax:
                                rx = -np.random.randint(xmax, xmin)
                            if ymin > ymax:
                                ry = -np.random.randint(ymax, ymin)

                            if xmin <= xmax:
                                rx = np.random.randint(xmin, xmax)
                            if ymin <= ymax:
                                ry = np.random.randint(ymin, ymax)

                            val = self.cal_dist(ay, ax, ry, rx)
                            if val < dbest:
                                xbest, ybest, dbest = rx, ry, val

                        except Exception as e:
                            print(e)
                            print(rand_d)
                            print(xmin, xmax)
                            print(ymin, ymax)
                            print(xbest, ybest)
                            print(self.B.shape)

                        rand_d = rand_d // 2

                    self.nnf[ay, ax] = [xbest, ybest]
                    self.nnd[ay, ax] = dbest

                    ax += xchange
                ay += ychange
            print("Done iteration {}".format(it + 1))
        print("Done All Iterations")
