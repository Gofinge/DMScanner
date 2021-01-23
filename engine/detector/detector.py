from data import DMImage
from utils.registry import Registry
from .gradient_edge_detector import dm_gradient_edge_detector
from .dev_detector import dm_dev_detector


class DMDetector:
    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self.output_dir = output_dir
        self.DETECTOR = Registry()
        self.DETECTOR.register("gradient-edge", dm_gradient_edge_detector)
        self.DETECTOR.register("dev", dm_dev_detector)

    def detect(self, dm_image: DMImage):
        return self.DETECTOR[self.cfg.METHOD](dm_image=dm_image, output_dir=self.output_dir, **self.cfg.ARG)
