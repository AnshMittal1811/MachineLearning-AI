import sys
sys.path.append("../")
sys.path.append("./")
from HtmlGenerator import HtmlGenerator


def main():
    # Gather a few data to display

    # A curve
    curve_data = {"loss": [1, 2, 3, 5]}

    # A parameter dict
    mydict = {
        "key1": 0,
        "key2": 1,
        "key3": [5, 6, 7],
        "key4": 3.141592,
        "key5": "toto",
        "key6": {"toto": 1, "tata": 2},
    }

    # A random matrix
    import numpy as np
    rows = 10
    cols = 6
    rand_matrix = np.random.randint(-50, 50, (rows, cols)) / 5.0

    # Let's make a website!
    webpage = HtmlGenerator(path="test/test.html", local_copy=True)

    # Make a title and a subtitle
    webpage.add_title("Table 1")
    webpage.add_subtitle("This is a subtitle")

    # Make a 1st table
    table1 = webpage.add_table("My awesome table")
    table1.add_column("Accuracy1")
    table1.add_column("Accuracy2")
    table1.add_column("Curve")
    table1.add_column("Mesh")
    curve = webpage.chart(curve_data, title="My curve")
    table1.add_row([0.5, curve, curve, webpage.mesh("test/test.obj", normalize=True)], "line1")
    webpage.return_html()

    # Make a 2nd table
    webpage.add_title("Table 2")
    table2 = webpage.add_table("Table_test")
    table2.add_columns(["Column1", "Column2"])
    table2.add_titleless_columns(1)

    table2.add_row(["data1", webpage.image("test/lena.jpeg"), webpage.dict(mydict), 0])
    table2.add_row([{"data1": 0.5}, webpage.image("test/lena.jpeg"), curve, webpage.confMat(rand_matrix)])
    table2.add_row(["Additional_data"]*9)
    table2.add_row(["Additional_data"]*9)
    table2.add_row([webpage.image("test/lena.jpeg")]*2)

    # curve_2 = webpage.curve(curve_data, title="My curve", width_factor=0.8)
    table2.add_row([webpage.chart({"data": [{'x':2, 'y':0}, {'x':3, 'y':0}, {'x':4, 'y':3}, {'x':10, 'y':0}]}, title="My curve", chart_type="scatter", width_factor=0.6),
                    webpage.chart(curve_data, title="My curve", width_factor=0.6, chart_type="bar"),
                    webpage.chart(curve_data, title="My curve", width_factor=0.6, chart_type="pie")])

    webpage.return_html(save_editable_version=True)

    webpage_after = HtmlGenerator(path="test/test2.html", reload_path="test/test.pkl")
    webpage_after.tables["Table_test"].add_row(["after edit test", "after edit test", "after edit test"])
    webpage_after.tables["My awesome table"].add_row(["after edit test", "after edit test", "after edit test", "after edit test"])
    webpage_after.add_title("test")
    webpage_after.return_html()

if __name__ == "__main__":
    main()
