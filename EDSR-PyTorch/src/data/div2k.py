import os
from data import srdata

class DIV2K(srdata.SRData):
    def __init__(self, args, name='DIV2K', train=True, benchmark=False):
        self.args = args
        data_range = [r.split('-') for r in args.data_range.split('/')]
        # print('data range: {}'.format(data_range))
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        # print('data range: {}'.format(data_range))
        # print('begin: {}, end: {}'.format(self.begin, self.end))
        super(DIV2K, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(DIV2K, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        # print('names_hr')
        # print(len(names_hr))
        # print('')
        # print('names_lr')
        # print(len(names_lr))
        # quit()
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        # print('dir_data={}'.format(dir_data))
        super(DIV2K, self)._set_filesystem(dir_data)
        if self.args.div2kvalid:
            self.dir_hr = os.path.join(self.apath, 'DIV2K_valid_HR')
            self.dir_lr = os.path.join(self.apath, 'DIV2K_valid_LR_bicubic')
        else:
            self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
            self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        # print('hr_dir={}, lr_dir={}'.format(len(self.dir_hr), len(self.dir_lr)))
        if self.input_large: self.dir_lr += 'L'

