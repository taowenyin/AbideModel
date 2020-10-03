from torch.utils import data


class AbideLacData(data.Dataset):
    def __getitem__(self, index: int):
        return self.pm_x[index], self.gm_x[index], self.sm_x[index], self.data_y[index]

    def __len__(self) -> int:
        return len(self.pm_x)

    def __init__(self, pm_x, gm_x, sm_x, data_y):
        self.pm_x = pm_x
        self.gm_x = gm_x
        self.sm_x = sm_x
        self.data_y = data_y

    def get_feature_size(self):
        return self.pm_x.shape[2]
