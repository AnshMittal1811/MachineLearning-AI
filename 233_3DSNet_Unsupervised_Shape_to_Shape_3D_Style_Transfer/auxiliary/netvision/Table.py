from ChartGenerator import Chart


class TableException(Exception):
    pass


class Table:
    def __init__(self, title):
        self.title = title
        self.columns = []  # 1st column contains the title
        self.rows = []
        self.are_columns_fixed = False
        self.td_str = None
        self.td_str_bold = None
        self.td_row_title_str = None
        self.width_percentage = None
        self.row_title_weight = 5.

    def add_column(self, title):
        if self.are_columns_fixed:
            raise TableException("You can't add a column after having added the first row!")
        self.columns.append(title)

    def add_columns(self, titles):
        if self.are_columns_fixed:
            raise TableException("You can't add a column after having added the first row!")
        self.columns += titles

    def add_titleless_columns(self, number):
        if self.are_columns_fixed:
            raise TableException("You can't add a column after having added the first row!")
        self.columns += [str(i + len(self.columns) + 1) for i in range(number)]

    def add_row(self, data, title=""):
        self.are_columns_fixed = True
        nb_rows_to_add = len(data) // len(self.columns)
        padding = (len(self.columns) - len(data) % len(self.columns)) % len(self.columns)
        if len(data) % len(self.columns) != 0:
            nb_rows_to_add += 1
        for i in range(nb_rows_to_add):
            title_row = str(len(self.rows) + 1) if title == "" else title
            self._make_td_str()
            row_str = f"<tr>\n{self.td_row_title_str}{title_row}</td>\n{self.td_str}"
            row_data = data[i * len(self.columns):(i + 1) * len(self.columns)]
            if i == nb_rows_to_add - 1:
                row_data += [""] * padding
            row_str += f"</td>\n{self.td_str}".join([str(self._pretreat_data(x)) for x in row_data])
            row_str += "</td>\n</tr>\n"
            self.rows.append(row_str)

    def _pretreat_data(self, data):
        if type(data) is Chart:
            data.width = f"({data.width_factor*self.width_percentage*self.row_title_weight}" \
                f"*window.innerWidth*0.01).toString() + \"px\""
        return data

    def _make_td_str(self):
        self.width_percentage = 100. / float(len(self.columns) * self.row_title_weight + 1)
        self.td_str = f"<td align=\"center\" width=\"{self.row_title_weight * self.width_percentage}%\">"
        self.td_row_title_str = f"<td style=\"font-weight: bold;\" align=\"center\" width=\"{self.width_percentage}%\">"
        self.td_str_bold = f"<td style=\"font-weight: bold;\" align=\"center\" width=\"{self.width_percentage}%\">"

    def __str__(self):
        self._make_td_str()
        data = "<table width=100%>\n"
        # Columns titles
        data += f"<tr>\n{self.td_row_title_str}{self.title}</td>\n{self.td_str_bold}"
        data += f"</td>\n{self.td_str_bold}".join([x for x in self.columns])
        data += "</td>\n</tr>\n"
        # Rows
        data += "\n".join(self.rows)
        data += "</table>\n"
        return data
