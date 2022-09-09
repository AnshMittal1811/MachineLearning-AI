import ChartGenerator
import MeshGenerator
from MeshGenerator import Mesh
import Table
import ConfusionMatrixGenerator
from os.path import abspath
import pickle
from shutil import copy
from os.path import join, exists, splitext
from os import makedirs

"""
TODO : 

Zoom curves
make minified version of javascript

Flag deploy
    -- add a zipping function of the whole sources
path de sortie = un dossier 'html'

Add barPlot

Gerer les pointclouds
Gerer les textures
Ajouter un curseur qui gere un mesh au cours du temps

Add minimized obj converter
Mesh generator should resize itself

todo : le jour ou j'ai que ca a foutre
-- bar de chargement
-- gerer les matrices (ou tensor) images numpy (sans avoir a les save quelque avant)
-- make a package pipy
"""


class HtmlGenerator:
    def __init__(self, path=None, title="NetVision visualization", reload_path=None, output_folder="media", local_copy = False):
        self.path = path
        self.head = []
        self.body = []
        self.curveGen = ChartGenerator.ChartGenerator()
        self.meshGen = MeshGenerator.MeshGenerator(html_path = abspath(self.path))
        self.confMatGen = ConfusionMatrixGenerator.ConfusionMatrixGenerator()
        self.tables = {}
        self.title = title
        self.hasCurveHeader = False
        self.hasMeshHeader = False
        self.hasDict_css = False
        self.make_header()
        self.make_body()
        self.local_copy = local_copy
        if reload_path is not None:
            with open(reload_path, 'rb') as file_handler:
                newObj = pickle.load(file_handler)
                self.__dict__.update(newObj.__dict__)
                self.path = path

        self.pict_it = 0

        self.output_folder = join('/'.join(path.split(sep='/')[:-1]), output_folder) # Output folder for the website, specified by the user
        self.image_folder = join(self.output_folder, "images")
        self.image_folder_relative_html = join(output_folder, "images")
        if not exists(self.image_folder):
            makedirs(self.image_folder)

    def make_header(self):
        self.head.append('<head>\n')
        self.head.append('\t<title></title>\n')
        self.head.append('\t<meta name=\"keywords\" content= \"Visual Result\" />  <meta charset=\"utf-8\" />\n')
        self.head.append('\t<meta name=\"robots\" content=\"index, follow\" />\n')
        self.head.append('\t<meta http-equiv=\"Content-Script-Type\" content=\"text/javascript\" />\n')
        self.head.append('\t<meta http-equiv=\"expires\" content=\"0\" />\n')
        self.head.append('\t<meta name=\"description\" content= \"Project page of style.css\" />\n')
        self.head.append('\t<link rel=\"shortcut icon\" href=\"favicon.ico\" />\n')
        self.head.append(" <style> .hor-bar { width:100%; background-color:black;  height:1px;   }"
                         " h3{  margin-top:10px; } </style>")

    def add_javascript_libraries(self):
        pass

    def add_css(self):
        pass

    def return_html(self, save_editable_version=False):
        # self.add_javascript_libraries()
        # self.add_css()

        begin_html = '<!DOCTYPE html>\n<html>\n'
        self.head_str = "".join(self.head)
        self.body_str = "".join([str(self._pretreat_data(x)) for x in self.body])

        end_html = "</html>\n"
        webpage = begin_html + self.head_str + self.meshGen.end_mesh() +  "</body>\n" + self.body_str + '</head>\n' + end_html
        if self.path is not None:
            with open(self.path, 'w') as output_file:
                output_file.write(webpage)
            if save_editable_version:
                with open(self.path[:-4] + "pkl", 'wb') as output_file:
                    pickle.dump(self, output_file)
        return webpage

    def _pretreat_data(self, data):
        if type(data) is ChartGenerator.Chart:
            data.width = f"(window.innerWidth*{data.width_factor}).toString() + \"px\""
        return data

    def make_body(self):
        self.body.append('<body style=\"background-color: lightgrey;\">\n')
        self.body.append('<center>\n')
        self.body.append('\t<div class=\"blank\"></div>\n')
        self.body.append('\t<h1>\n')
        self.body.append(f'\t\t{self.title}\n')
        self.body.append('\t</h1>\n')
        self.body.append('</center>\n')
        self.body.append('<div class=\"blank\"></div>\n')

    def add_html_in_body(self, html_content):
        """
        :param html_content: html string.
        :return:
        """
        self.body.append(html_content)

    def add_title(self, title_content):
        body = []
        body.append('\t<h2>\n')
        body.append(f'\t\t{title_content}\n')
        body.append('\t</h2>\n')
        self.body.append("".join(body))

    def add_subtitle(self, sub_title_content):
        body = []
        body.append('\t<h3>\n')
        body.append(f'\t\t{sub_title_content}\n')
        body.append('\t</h3>\n')
        self.body.append("".join(body))

    def add_subsubtitle(self, sub_title_content):
        body = []
        body.append('\t<h4>\n')
        body.append(f'\t\t{sub_title_content}\n')
        body.append('\t</h4>\n')
        self.body.append("".join(body))

    def add_linebreak(self):
        self.body.append(f'</br>\n')

    def image(self, path, size="300px"):
        if self.local_copy:
            in_pict_file = path  # path to the image
            pict_new_name = str(self.pict_it).zfill(3) + splitext(in_pict_file)[1]
            out_pict_file = join(self.image_folder, pict_new_name)
            copy(in_pict_file, out_pict_file)
            path = join(self.image_folder_relative_html, pict_new_name)  # Path to use in html code
            self.pict_it += 1

        body = []
        body.append(f'<a download={path} href={path} title="ImageName"> '
                    f'<img  src={path} width={size} height={size} /></a>\n')
        return "".join(body)

    def add_image(self, path, size="300px"):
        self.body.append(self.image(path, size))

    def chart(self, data, chart_type="line", title=None, x_labels=None, font_color="black", width_factor=1, ):
        if not self.hasCurveHeader:
            self.head.append(self.curveGen.make_header())
        self.hasCurveHeader = True
        return self.curveGen.make_chart(data=data, font_color=font_color, chart_type=chart_type,
                                        title=title, width_factor=width_factor, x_labels=x_labels)

    def add_chart(self, data, chart_type="line", title=None, x_labels=None, font_color="black", width_factor=1):
        self.body.append("<div>")
        self.body.append(self.chart(data, chart_type=chart_type, title=title, x_labels=x_labels, font_color=font_color, width_factor=width_factor))
        self.body.append("</div>")

    def text(self, text):
        return text

    def add_textFile(self, path):
        self.body.append(f"<object  width=\"2000\" height=\"1000\"  type=\"text/plain\" data=\"{path}\" border=\"0\" ></object>")

    def mesh(self, mesh_path, title="", normalize=True):
        if not self.hasMeshHeader:
            self.head.append(self.meshGen.make_header())
        self.hasMeshHeader = True
        if self.local_copy:
            in_pict_file = mesh_path  # path to the image
            pict_new_name = str(self.pict_it).zfill(3) + splitext(in_pict_file)[1]
            out_pict_file = join(self.image_folder, pict_new_name)
            copy(in_pict_file, out_pict_file)
            mesh_path = join(self.image_folder_relative_html, pict_new_name)  # Path to use in html code
            self.pict_it += 1
            if normalize:
                Mesh(out_pict_file)            
        return self.meshGen.make_mesh(mesh_path, title)

    def add_table(self, title=""):
        table = Table.Table(title)
        self.body.append(table)
        if len(title) == 0:
            title = str(len(self.tables))
        self.tables[title] = table
        return table

    def confMat(self, data, rows_titles=None, colums_titles=None, title="Confusion", colormap=None):
        return self.confMatGen.make_confusionmatrix(data, rows_titles, colums_titles, title=title, colormap=colormap)

    def dict(self, data, title="PARAMETERS"):
        if not self.hasDict_css:
            self.head.append(self.add_css_for_add_dict())
        self.hasDict_css = True
        out_string = f"<span class=\"value\">{title} </span></br>\n"
        for key in data.keys():
            out_string += f"<span class=\"key\"> {key} </span> : <span class=\"value\">{data[key]} </span></br>\n"
        return out_string

    def add_css_for_add_dict(self):
        outstring = ""
        outstring += "<style>\n\
              .key {\n\
                color: #2980b9;\n\
                font-weight:bold; \n\
              }\n\
              .value { /* OK, a bit contrived... */\n\
                color: #c0392b;\n\
                font-weight:bold; \n\
                }</style>\n\
            "
        return outstring

    def dump(self):
        pass


if __name__ == '__main__':
    import numpy as np

    webpage = HtmlGenerator(path="test/test.html")
    mydict = {
        "key1": 0,
        "key2": 1,
        "key3": [5, 6, 7],
        "key4": np.pi,
        "key5": "toto",
        "key6": {"toto": 1, "tata": 2},
    }
    webpage.add_html_in_body(webpage.add_dict(mydict))
    rows = 20
    cols = 22
    rand_matrix = np.random.randint(-50, 50, (rows, cols)) / 5.0
    # webpage.add_html_in_body(webpage.mesh("test/output_atlas.obj"))
    #
    # import matplotlib.pyplot as plt
    # colormap = plt.get_cmap("nipy_spectral")
    # webpage.add_html_in_body(webpage.add_confMat(rand_matrix, colormap=colormap))
    #
    # webpage.add_html_in_body(webpage.mesh("test/output_atlas.obj"))
    webpage.return_html()
