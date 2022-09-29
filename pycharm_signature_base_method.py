
class Base:
  """foo"""

  def method(self, arg1, **kwargs):
    """bar"""

  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, **kwargs):
    """x"""


class Obj(Base):
  """foo"""

  def method(self, arg1, arg2, **kwargs):
    """bar"""

  @classmethod
  def get_rec_initial_extra_outputs(cls, batch_dim, rec_layer, *, network, num_heads, total_key_dim, n_out, name,
                                    initial_state=None, sources=(), **kwargs):
    """y"""
