from functools import wraps
import numpy as np
import matplotlib.pyplot as plt

def output_q_vals(q_func):
    @wraps(q_func)
    def _q_func(self, x, test=False):
        action_val = q_func(self, x, test)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        q_vals = action_val.q_values.data[0].reshape(8, 8)
        ax.imshow(q_vals, vmin=-1, vmax=1)

        y_, x_ = np.where(x[0, 0:64].reshape(8, 8).astype(np.bool))
        ax.plot(x_, y_, 'o', color='g')
        board = x[0, 64:].reshape(8, 8, 3).astype(np.bool)
        y_, x_ = np.where(board[:, :, 0])
        ax.plot(x_, y_, 'o', color='r')
        y_, x_ = np.where(board[:, :, 1])
        ax.plot(x_, y_, 'o', color='b')

        action = action_val.greedy_actions.data[0]
        y_, x_ = divmod(action, 8)
        ax.plot(x_, y_, 'o', color='m')

        for i in range(8):
            for j in range(8):
                ax.text(j, i + 0.5, "{:.2f}".format(q_vals[i, j]),
                    ha='center', va='bottom', color='w', alpha=1.0)

        if False:
            plt.show()
        else:
            plt.savefig("img/q_vals-{}.png".format(_q_func._img_idx))
            _q_func._img_idx += 1

        return action_val

    _q_func._img_idx = 0
    return _q_func
