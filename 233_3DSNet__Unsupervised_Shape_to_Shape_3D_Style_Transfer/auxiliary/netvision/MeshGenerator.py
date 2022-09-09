from os.path import join, dirname, relpath, abspath
from os import getcwd
import numpy as np

class Mesh(object):
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.read_obj()
        self.normalize()
        self.write_obj()

    def read_obj(self):
        with open(self.obj_path, "r") as in_file:
            points = []
            faces = []
            for line in in_file:
                line_s = line.split()
                if len(line_s)==0:
                    continue
                if line_s[0] == "v":
                    points.append([float(x) for x in line_s[1:4]])
                if line_s[0] == "f":
                    faces.append([int(x.split(sep='/')[0]) - 1 for x in line_s[1:4]])
        
        self.points = points
        self.faces = faces

    def normalize(self):
        centroid = np.mean(self.points, axis=0, keepdims=True)
        self.points = self.points - centroid
        self.points = self.points / np.sqrt(np.max(np.sum(self.points**2, 1)))

    def write_obj(self):
        with open(self.obj_path, "w") as out_file:
            out_file.write("\n".join(["v " + " ".join([str(coord) for coord in point]) for point in
                                      self.points])
                           + "\n"
                           + "\n".join(["f " + " ".join([str(int(tri + 1)) for tri in face]) for face in
                                        self.faces]))

class MeshGenerator:
    def __init__(self, html_path):
        self.curve_it = 0
        self.colors = ["#c0392b", " #2980b9", "#27ae60"]
        self.three_path = join(dirname(__file__), "js/three.js")
        self.Detector_path = join(dirname(__file__), "js/Detector.js")
        self.OrbitControls_path = join(dirname(__file__), "js/OrbitControls.js")
        self.OBJLoader_path = join(dirname(__file__), "js/OBJLoader.js")
        self.MTLLoader_path = join(dirname(__file__), "js/MTLLoader.js")
        self.event_listener = []
        self.added_mesh = []
        self.html_path = html_path


    def make_header(self):
        ret_str = ""
        js_libs = [self.three_path, self.Detector_path, self.OrbitControls_path, self.OBJLoader_path,
                   self.MTLLoader_path]
        print(js_libs)
        for file in js_libs:
            with open(file, "r") as js_file:
                print(file)
                ret_str += "  <script type=\"text/javascript\">\n  " + js_file.read().replace("\n",
                                                                                              "\n  ") + " \n</script>\n"
        return ret_str

    def make_mesh(self, mesh_path, title=None):

        # mesh_path = abspath(join(getcwd(), mesh_path))
        # mesh_path = relpath(mesh_path, dirname(self.html_path))
        out_string = f"<div id=\"mesh_{self.curve_it}\"> <h4>{title}</h4> </div>\n"

        out_string += "     <script>\n"
        out_string += "     if (!Detector.webgl) {\nDetector.addGetWebGLMessage();\n}\n"
        init_function = ""
        init_function += "     var my_mesh;\nvar objLoader_my_mesh;\n"
        init_function += "     var camera_my_mesh, controls_my_mesh, scene_my_mesh, renderer_my_mesh;\n"
        init_function += "     var lighting, ambient, keyLight_my_mesh, fillLight_my_mesh, backLight_my_mesh;\n"
        init_function += "     var windowX = 400;\nvar windowY = 250;\n"
        init_function += "     init_my_mesh();\nanimate_my_mesh();\n"

        init_function += "\n\
            function init_my_mesh() {\n\
                my_mesh = document.getElementById('my_mesh');\n\
                /* Camera */\n\
                camera_my_mesh = new THREE.PerspectiveCamera(10, 1, 0.1, 2500);\n\
                camera_my_mesh.position.set( 5, 5, -10 );\n\
                /* Scene */\n\
                scene_my_mesh = new THREE.Scene();\n\
                lighting = true;\n\
                ambient_my_mesh = new THREE.AmbientLight(0xffffff, 0.15);\n\
                scene_my_mesh.add(ambient_my_mesh);\n\
                keyLight_my_mesh = new THREE.DirectionalLight(0xffffff, 0.15);\n\
                keyLight_my_mesh.position.set(-100, 0, 100);\n\
                fillLight_my_mesh = new THREE.DirectionalLight(0xffffff, 0.15);\n\
                fillLight_my_mesh.position.set(100, 0, 100);\n\
                fillLight1_my_mesh = new THREE.DirectionalLight(0xffffff, 0.15);\n\
                fillLight2_my_mesh = new THREE.DirectionalLight(0xffffff, 0.15);\n\
                fillLight3_my_mesh = new THREE.DirectionalLight(0xffffff, 0.15);\n\
                fillLight4_my_mesh = new THREE.DirectionalLight(0xffffff, 0.15);\n\
                backLight_my_mesh = new THREE.DirectionalLight(0xffffff, 1.0);\n\
                backLight_my_mesh.position.set(100, 0, -100).normalize();\n\
                scene_my_mesh.position.set( -0.25,-0.25,0 );\n\
                scene_my_mesh.add(keyLight_my_mesh, fillLight_my_mesh, backLight_my_mesh , fillLight1_my_mesh, fillLight2_my_mesh, fillLight3_my_mesh, fillLight4_my_mesh);\n\
                /* Model */\n\
                objLoader_my_mesh = new THREE.OBJLoader();\n\
                objLoader_my_mesh.load('output_atlas.obj', function (object) {\n\
                    object.name = 'object';\n\
                    scene_my_mesh.add(object);\n\
                });\n\
                /* Renderer */\n\
                renderer_my_mesh = new THREE.WebGLRenderer();\n\
                renderer_my_mesh.setPixelRatio(window.devicePixelRatio);\n\
                renderer_my_mesh.setSize(windowX, windowY);\n\
                renderer_my_mesh.setClearColor(new THREE.Color(\"#D3D3D3\"));\n\
                my_mesh.appendChild(renderer_my_mesh.domElement);\n\
                /* Controls */\n\
                controls_my_mesh = new THREE.OrbitControls(camera_my_mesh, renderer_my_mesh.domElement);\n\
                controls_my_mesh.enableDamping = true;\n\
                controls_my_mesh.dampingFactor = 0.25;\n\
                controls_my_mesh.enableZoom = false;\n\
                /* Events */\n\
                }\
                "

        init_function += "\n\
            function animate_my_mesh() {\n\
                requestAnimationFrame(animate_my_mesh);\n\
                controls_my_mesh.update();\n\
                render_my_mesh();\n\
                }\n\
            "

        init_function += "\n\
            function render_my_mesh() {\n\
                renderer_my_mesh.render(scene_my_mesh, camera_my_mesh);\n\
                }\n\
             "

        init_function = init_function.replace("my_mesh", "mesh_" + str(self.curve_it)).replace('output_atlas.obj', mesh_path)
        self.added_mesh.append("mesh_" + str(self.curve_it))
        out_string += init_function

        out_string += "</script>\n"

        self.curve_it += 1

        return out_string

    def make_onWindowResize(self):
        self.event_listener.append("\
            window.addEventListener('resize', onWindowResize, false);\n\
            window.addEventListener('keydown', onKeyboardEvent, false);\n\
            \n\
            ")

        onWindowResize = "\n\
                function onWindowResize() {\n\
                 "

        for mesh_id in self.added_mesh:
            local_str = f"camera_my_mesh.aspect = windowX / windowY;\n\
                    camera_my_mesh.updateProjectionMatrix();\n\
                    renderer_my_mesh.setSize(windowX, windowY);\n "
            onWindowResize += local_str.replace("my_mesh", mesh_id)
        onWindowResize += " }\n "
        self.event_listener.append(onWindowResize)

    def make_init_function(self):
        init_function = "\n\
            function onKeyboardEvent(e) {\n\
                if (e.code === 'KeyL') {\n\
                    lighting = !lighting;\n\
                    if (lighting) {\n\
                "

        for mesh_id in self.added_mesh:
            local_str = "ambient_my_mesh.intensity = 0.25;\n\
                        scene_my_mesh.add(keyLight_my_mesh);\n\
                        scene_my_mesh.add(fillLight_my_mesh);\n\
                        scene_my_mesh.add(fillLight1_my_mesh);\n\
                        scene_my_mesh.add(fillLight2_my_mesh);\n\
                        scene_my_mesh.add(fillLight3_my_mesh);\n\
                        scene_my_mesh.add(fillLight4_my_mesh);\n\
                        scene_my_mesh.add(backLight_my_mesh);\n"
            init_function += local_str.replace("my_mesh", mesh_id)

        init_function += " } else {\n\
                    "

        for mesh_id in self.added_mesh:
            local_str = "ambient_my_mesh.intensity = 1.0;\n\
                        scene_my_mesh.remove(keyLight_my_mesh);\n\
                        scene_my_mesh.remove(fillLight_my_mesh);\n\
                        scene_my_mesh.remove(fillLight1_my_mesh);\n\
                        scene_my_mesh.remove(fillLight2_my_mesh);\n\
                        scene_my_mesh.remove(fillLight3_my_mesh);\n\
                        scene_my_mesh.remove(fillLight4_my_mesh);\n\
                        scene_my_mesh.remove(backLight_my_mesh);\n"
            init_function += local_str.replace("my_mesh", mesh_id)

        init_function += "        }\n\
                }\n\
                }\n\
            "
        self.event_listener.append(init_function)

    def end_mesh(self):
        """
        This function is safe to call as many time as one wants
        :return:
        """
        self.event_listener = []
        self.event_listener.append("<script>\n")
        self.make_onWindowResize()
        self.make_init_function()
        self.event_listener.append("</script>\n")
        return "".join(self.event_listener)
