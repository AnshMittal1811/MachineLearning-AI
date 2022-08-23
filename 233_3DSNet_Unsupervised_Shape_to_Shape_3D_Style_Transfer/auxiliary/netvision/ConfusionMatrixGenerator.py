import numpy as np
from os.path import join, dirname


class ColorMap:
    def __init__(self):
        self.colormap_path = join(dirname(__file__), "colormap.npy")
        self.colormap = np.load(self.colormap_path)

    def __call__(self, index):
        """
        :param value: a float
        :return:
        """
        colors = self.colormap[index]
        return colors


class ConfusionMatrixGenerator:
    def __init__(self):
        self.colormap = ColorMap()

    def update_colormap(self, colormap):
        self.colormap = colormap

    def make_confusionmatrix(self, data, rows_titles, colums_titles, title="Confusion", colormap=None):
        if colormap is not None:
            self.update_colormap(colormap)

        if rows_titles is None:
            rows_titles = np.arange(data.shape[0])
        if colums_titles is None:
            colums_titles = np.arange(data.shape[1])

        outstr = ""
        outstr += "<table cellspacing=\"0\" cellpadding=\"2px\">\n"
        data_scaled = self.rescale(data)

        outstr += "<tr>\n"
        outstr += f"<td  style=\"text-align: center; font-weight: bold;\" >{title}</td>\n"

        for j in range(data.shape[1]):
            outstr += f"<td  style=\"text-align: center; font-weight: bold;\" > {colums_titles[j]} </td>\n"
        outstr += f"<td  style=\"text-align: center; font-weight: bold;\" ></td>\n"
        outstr += f"<td  style=\"text-align: center; font-weight: bold;\" >Colormap</td>\n"
        outstr += "</tr\n>"

        for i in range(data.shape[0]):
            outstr += f"<tr>\n"
            outstr += f"<td style=\"text-align: center; font-weight: bold;\"> {rows_titles[i]} </td>\n"

            for j in range(data.shape[1]):
                rgb_float = self.colormap(int(data_scaled[i, j]))[:3]
                rgb_int = self.rgb(rgb_float)
                color_str = f"rgb({rgb_int[0]}, {rgb_int[1]}, {rgb_int[2]})"
                outstr += f"<td style=\"color: white ; text-align: center; background-color:{color_str}\"> {data[i, j]} </td>\n"

            outstr += f"<td  style=\"text-align: center; font-weight: bold;\" ></td>\n"

            index = int((float(i) / float(data.shape[0])) * 255.0)
            index_rescaled = int((float(i) / float(data.shape[0])) * (np.max(data) - np.min(data)) + np.min(data))
            rgb_float = self.colormap(index)[:3]
            rgb_int = self.rgb(rgb_float)
            color_str = f"rgb({rgb_int[0]}, {rgb_int[1]}, {rgb_int[2]})"
            outstr += f"<td style=\"color: white ; text-align: center; background-color:{color_str}\"> {index_rescaled} </td>\n"

            outstr += "</tr>\n"
        outstr += "</table>\n"
        return outstr

    def make_header(self):
        pass

    def rescale(self, data):
        "rescale to 0-255"
        data = data - np.min(data)
        data = data / np.max(data)
        data = data * 255
        data = np.floor(data)
        return data

    def rgb(self, color):
        "rescale to 0-255"
        color = [int(np.floor(data * 255)) for data in color]
        return color

    def end_confusionmatrix(self):
        pass
