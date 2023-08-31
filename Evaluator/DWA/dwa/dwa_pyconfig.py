class RobotProp:
    """机器人参数"""
    def __init__(self):
        self.radius = 0.17
        self.maxV = 1.0
        self.minV = 0
        self.maxW = 0.9
        self.minW = -0.9
        self.accV = 5.0
        self.accW = 4.0

        self.resolV = 0.1
        self.resolW = 0.1
        self.dt     = 0.25
        self.T      = 1.0
        self.kalpha = 0.6
        self.kro = 3.0
        self.kv  = 1.0
        # DWA cost 系数
        self.weightV = 0.05
        self.weightG = 1.0
        self.weightObs = 0.5
