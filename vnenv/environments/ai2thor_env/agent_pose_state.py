from typing import Dict, Optional
DEFAULT_Y = 0.91  # THOR环境是平坦的，因此智能体的高为一个定植
# 默认为z轴为旋转参考轴，y轴为高度轴


class AgentPoseState:
    """表示智能体在离散THOR环境中的位置姿态的类"""

    def __init__(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        rotation: float = 0,
        horizon: float = 0,
        pose_str: Optional[str] = None
    ) -> None:
        if pose_str is not None:
            x, z, rotation, horizon = [float(x) for x in pose_str.split("|")]
            y = DEFAULT_Y
        self.x = round(x, 2)
        self.y = round(y, 2)
        self.z = round(z, 2)
        self.rotation = round(rotation)
        self.horizon = round(horizon)

    def __eq__(self, other) -> bool:
        """比较两个位姿是否相同"""
        if isinstance(other, AgentPoseState):
            return (
                self.x == other.x
                and
                # thor中y值一定相同
                # self.y == other.y and
                self.z == other.z
                and self.rotation == other.rotation
                and self.horizon == other.horizon
            )

    def __str__(self) -> str:
        """返回字符串形式的智能体位姿状态, x与z保留两位小数
        """
        return "{:0.2f}|{:0.2f}|{:d}|{:d}".format(
            self.x, self.z, round(self.rotation), round(self.horizon)
        )

    def position(self) -> Dict[str, float]:
        """只返回坐标"""
        return dict(x=self.x, y=self.y, z=self.z)
