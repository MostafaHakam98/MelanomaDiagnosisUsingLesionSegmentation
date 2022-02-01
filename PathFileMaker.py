import os

class fileMaker:

    def __init__(self, textfile_train="", textfile_valid="", path_images_train="", path_images_valid=""):
        self.textfile_train = textfile_train
        self.textfile_valid = textfile_valid
        self.path_images_train = path_images_train
        self.path_images_valid = path_images_valid
        self.create_list_files()

    def create_list_files(self):
        a = open(self.textfile_train, "w")

        for path, subdirs, files in os.walk(self.path_images_train):
            for file in files:
                f = os.path.join(file)
                if str(f)[-4:] == ".jpg":
                    a.write(str(f) + os.linesep)

        if self.textfile_valid:
            b = open(self.textfile_valid, "w")

            for path, subdirs, files in os.walk(self.path_images_valid):
                for file in files:
                    f = os.path.join(file)
                    if str(f)[-4:] == ".jpg":
                        b.write(str(f) + os.linesep)
