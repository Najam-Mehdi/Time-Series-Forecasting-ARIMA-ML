def data_review(df):
    info = {
        'Number of Rows': df.shape[0],
        'Number of Columns': df.shape[1],
        'Missing Values': df.isnull().sum().sum(),
        'Duplicate Values': df.duplicated().sum(),
    }
    return info


def plot_series(series, array_like, label='', title=''):
    import matplotlib.pyplot as plt
    data_range = range(len(series))
    plt.plot(data_range, array_like, label=label)
    plt.title(title)
    plt.legend()
    plt.grid(True)