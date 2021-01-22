import os
from .DMImage import DMImage


class DMDataset:
    def __init__(self, source_dir, cfg):
        self.cfg = cfg
        self.source_dir = source_dir
        self.image_list = []
        self.extension_write_list = [".png", ".bmp", ".jpg"]

        for root, _, files in os.walk(self.source_dir):
            for file in files:
                _, extension = os.path.splitext(file)
                if extension in self.extension_write_list:
                    self.image_list.append(os.path.join(root, file))

    def __getitem__(self, key):
        return DMImage(self.image_list[key], self.cfg)

    def __len__(self):
        return len(self.image_list)
