
# run: pylint pylint-astroid-bug1452.py

# https://github.com/PyCQA/astroid/issues/437
# https://github.com/PyCQA/pylint/issues/1452
# https://github.com/rwth-i6/returnn/blob/master/TFUtil.py#L559


class Data:
    def __init__(self):
        self.shape = [None, 17]

    @property
    def ndim(self):
        return len(self.shape)

    def copy_move_axis(self, old_axis):
        if old_axis < 0:
            old_axis += self.ndim
            assert old_axis >= 0
        assert 0 <= old_axis < self.ndim

        new_shape = [None] * self.ndim
        self.shape = new_shape
        return self
