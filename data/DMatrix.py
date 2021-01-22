class DMatrix:
    """
        The class to define a datamatrix code.
    """

    def __init__(self):
        # polygon: 目前暂不考虑多于四个点的polygon。
        # 0 .. 3
        # |    :
        # 1 -- 2
        self.polygon = []
        self.bounding_rect = []
        self.decode_data = None
        self.height = None
        self.width = None
        self.type = None  # default: ECC200.