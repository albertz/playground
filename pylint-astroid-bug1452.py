
# run: pylint pylint-astroid-bug1452.py

# https://github.com/PyCQA/astroid/issues/437
# https://github.com/PyCQA/pylint/issues/1452
# https://github.com/rwth-i6/returnn/blob/master/TFUtil.py#L559


class Data:
    def __init__(self):
        self.shape = [None, 17]
        self.batch_dim_axis = 0

    def copy(self):
        return Data()

    @property
    def ndim(self):
        """
        :rtype: int
        :return: ndim counted without batch-dim
        """
        return len(self.shape)

    @property
    def batch_ndim(self):
        """
        :rtype: int
        :return: ndim counted with batch-dim
        """
        if self.batch_dim_axis is not None:
            return self.ndim + 1
        return self.ndim

    def copy_move_axis(self, old_axis, new_axis):
        """
        :param int old_axis: counted with batch-dim
        :param int new_axis: counted with batch-dim
        :return: copy of myself with moved axis (see :func:`move_axis`)
        :rtype: Data
        """
        if old_axis < 0:
            old_axis += self.batch_ndim
            assert old_axis >= 0
        assert 0 <= old_axis < self.batch_ndim
        if new_axis < 0:
            new_axis += self.batch_ndim
            assert new_axis >= 0
        assert 0 <= new_axis < self.batch_ndim
        if old_axis == new_axis:
            return self.copy()

        data = self.copy()
        new_shape = [None] * data.ndim
        data.shape = tuple(new_shape)
        return data
