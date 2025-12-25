class Recorder:
    def __init__(self, file=None, folder=None):
        self.iter = 0
        self.file = file
        self.folder = folder
        pass

    def record(self, content, append_on_file=True):
        if append_on_file:
            assert self.file is not None
            tar_file = self.file
        else:
            assert self.folder is not None
            tar_file = f'{self.folder}/{self.iter}.txt'

        self.iter += 1
        f = open(tar_file, 'a+')
        f.write(content)
        f.close()
