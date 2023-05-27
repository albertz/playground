
import dataclasses


@dataclasses.dataclass
class F:
    """
    Attributes:
        x: foo
    """

    x: int

    y: int = 0
    "bar"


f = F(1)
print(f.x)
print(f.y)
