import os
import re
import sys
import copy
import torch
import numpy as np
from PIL import Image
from . import show, labwidget

stylegan_path = './models/networks/stylegan3'
sys.path.append(stylegan_path)
import legacy
import dnnlib


##########################################################################
# UI
##########################################################################

export_text = labwidget.Property(f"Welcome to GAN Warping!")


class GANWarpingApp(labwidget.Widget):
    def __init__(self, modelbase, modeldir, savedir='exported_weights', size=256, num_samples=9, num_pages=5, seed=3000, trunc=1.0, icon_seed=6000, device='cuda'):
        super().__init__(className='rwa',
                         style={
                             'border': '0',
                             'padding': '0',
                             'display': 'inline-block',
                             'width': '1100px',
                             'left': '0',
                             'margin': '0',
                             'background-color': '#4b4b4b',
                         })
        self.size = size
        self.trunc = trunc
        self.device = device
        self.orig_gen = self.get_gen_model(modelbase, device)
        self.curr_gen = self.orig_gen
        self.model_zoo = self.collect_model_weights(modeldir, device)

        os.makedirs(savedir, exist_ok=True)
        self.savedir = savedir

        self.curr_page = 0
        self.num_samples = num_samples
        self.num_pages = num_pages
        self.all_noise_z = self.get_noise_z(num_samples * num_pages, self.orig_gen.z_dim, seed, device)
        self.noise_z = self.all_noise_z[:num_samples]
        self.icon_z = self.get_noise_z(1, self.orig_gen.z_dim, icon_seed, device)

        reset_sty = dict(margin='auto', background='#AA00CF')
        sample_sty = {'display': 'inline', 'background': '#0088DD', 'vertical-align': 'bottom', 'margin-left': 'auto', 'margin-right': '0'}
        export_sty = {'display': 'inline', 'background': '#DD8800', 'vertical-align': 'bottom', 'margin-left': 'auto', 'margin-right': '0'}
        self.reset_btn = labwidget.Button('Reset', reset_sty).on('click', self.reset_sliders)
        self.sample_btn = labwidget.Button('More samples', style=sample_sty).on('click', self.change_noise)
        self.export_btn = labwidget.Button('Export', style=export_sty).on('click', self.export_model)

        self.export_status = labwidget.Div(style={'font-size': '16px', 'color': 'gray', 'float': 'left', 'display': 'inline-block'})
        self.text_label = labwidget.Label(export_text)
        self.export_status.show([self.text_label])

        self.image_array = []
        for i in range(num_samples):
            self.image_array.append(
                labwidget.Image(style={'width': f'{size}px', 'height': f'{size}px'})
            )
        self.refresh_images()
        self.sidebar = self.create_all_sliders()

    def get_gen_model(self, modelbase, device):
        with dnnlib.util.open_url(modelbase) as fp:
            G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)  # type: ignore
        return G

    def collect_model_weights(self, modeldir, device):
        weight_paths = sorted([os.path.join(modeldir, path) for path in os.listdir(modeldir) if path.endswith('.pth')])
        return [torch.load(p, map_location=device) for p in weight_paths]

    def export_model(self):
        prev_models = os.listdir(self.savedir)
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_models]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        save_path = os.path.join(self.savedir, f'{cur_run_id:05d}.pth')

        self.curr_gen.to('cpu')
        torch.save(self.curr_gen.state_dict(), save_path)
        self.curr_gen.to(self.device)

        export_text.set(f"Exported model to {save_path}")

    def get_noise_z(self, num_samples, z_dim, seed, device):
        noise_z = np.random.RandomState(seed).randn(num_samples, z_dim)
        return torch.from_numpy(noise_z).to(device)

    def change_noise(self):
        self.curr_page = (self.curr_page + 1) % self.num_pages
        start = self.curr_page * self.num_samples
        end = (self.curr_page + 1) * self.num_samples
        self.noise_z = self.all_noise_z[start:end]
        self.refresh_images()

    def reset_sliders(self):
        for idx in range(len(self.model_zoo)):
            # gather slider widgets and attributes
            icon = getattr(self, f'slider{idx}_icon')
            scale_range = getattr(self, f'slider{idx}_range')

            # update css styles and knob values
            setattr(self, f'slider{idx}_on', False)
            setattr(self, f'slider{idx}_scale', 1)
            icon.style = {'opacity': 0.2, 'width': '135px', 'height': '135px'}
            scale_range.style = {'display': 'inline', 'pointer-events': 'none', 'opacity': 0.2}

        self.update_model()

    @torch.no_grad()
    def refresh_images(self):
        images = self.curr_gen(self.noise_z, None, truncation_psi=self.trunc)
        images = (images + 1) * (255 / 2)
        images = images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
        for idx, img in enumerate(images):
            img_pil = Image.fromarray(img)
            self.image_array[idx].render(img_pil)

    @torch.no_grad()
    def update_model(self):
        edit_gen = copy.deepcopy(self.orig_gen)
        orig_state_dict = self.orig_gen.state_dict()
        edit_state_dict = edit_gen.state_dict()
        for idx, model in enumerate(self.model_zoo):
            # only plug new weights when the model is toggled on
            on = getattr(self, f'slider{idx}_on')
            if not on:
                continue
            # udpate the weights according to the scale
            scale = float(getattr(self, f'slider{idx}_scale'))
            for name, param in model.items():
                orig_w = orig_state_dict[name]
                edit_w = edit_state_dict[name]
                diff_w = param - orig_w
                new_w = edit_w + scale * diff_w
                edit_state_dict[name].copy_(new_w)

        # update the current generator and render images
        self.curr_gen = edit_gen
        self.refresh_images()

    @torch.no_grad()
    def create_all_sliders(self):
        sliders = []
        for idx, model in enumerate(self.model_zoo):
            # load weights for the edited generators
            edit_gen = copy.deepcopy(self.orig_gen)
            state_dict = edit_gen.state_dict()
            for name, param in model.items():
                state_dict[name].copy_(param)

            # create icons
            icon = edit_gen(self.icon_z, None, truncation_psi=self.trunc)[0]
            icon = ((icon + 1) * (255 / 2)).permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
            icon = Image.fromarray(icon)

            # setup individual slider for each model
            sliders.append(self.create_model_slider(icon, idx))

        sidebar = labwidget.Div(style={'margin': 'auto'})
        sidebar.show(sliders)
        return sidebar

    def create_model_slider(self, icon_image, idx):
        # set up the widgets for the slider
        slider = labwidget.Div()

        icon = labwidget.Image(style={'opacity': 0.2, 'width': '135px', 'height': '135px'}).on('click', self.toggle_model_from_idx(idx))
        icon.render(icon_image)

        scale_range_style = {'pointer-events': 'none', 'opacity': 0.2, 'width': '135px'}
        scale = labwidget.Property(1.0)
        scale_range = labwidget.Range(value=scale, min=0, max=1, step=0.01, style=scale_range_style)
        slider.show([icon, scale_range])

        # set all widget as attributes to the class
        setattr(self, f'slider{idx}', slider)
        setattr(self, f'slider{idx}_on', False)
        setattr(self, f'slider{idx}_scale', scale)
        setattr(self, f'slider{idx}_icon', icon)
        setattr(self, f'slider{idx}_range', scale_range)

        # listen to the scale changes to update the models
        self.on(f'slider{idx}_scale', self.update_model)

        return slider

    def toggle_model_from_idx(self, idx):
        def toggle_model(event):
            # gather slider widgets and attributes
            slider = getattr(self, f'slider{idx}')
            on = getattr(self, f'slider{idx}_on')
            icon = getattr(self, f'slider{idx}_icon')
            scale_range = getattr(self, f'slider{idx}_range')

            # update css styles and knob values
            if on:
                setattr(self, f'slider{idx}_on', False)
                icon.style = {'opacity': 0.2, 'width': '135px', 'height': '135px'}
                scale_range.style = {'pointer-events': 'none', 'opacity': 0.2, 'width': '135px'}
            else:
                setattr(self, f'slider{idx}_on', True)
                icon.style = {'width': '135px', 'height': '135px'}
                scale_range.style = {'width': '135px'}

            # udpate the slider div innerHTML
            slider.show([icon, scale_range])

            # update the new model and rendered images
            self.update_model()

        return toggle_model

    def widget_html(self):
        def h(w):
            return w._repr_html_()
        return f'''<div {self.std_attrs()}>
        <div>
        <style>
        .rwa input[type=button] {{ background: dimgray; color: white;
          border: 0; border-radius: 8px; padding: 5px 10px; font-size: 18px; }}
        .sidebar input[type=button] {{ background: #45009E; }}
        ::-webkit-scrollbar {{ width: 10px; }}
        ::-webkit-scrollbar-track {{ background: #888; }}
        ::-webkit-scrollbar-thumb {{ background: #b1b1b1; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #f1f1f1; }}
        </style>
        <center><span style="font-size:24px;margin-right:24px;margin-top:10px;color:white;vertical-align:bottom;">
        {h(self.export_status)}
        <br>GAN Warping
        {h(self.sample_btn)}
        {h(self.export_btn)}
        </span></center>
        <div style="margin-top: 8px; margin-bottom: 8px;"><!-- middle row -->
        <hr style="border:1px solid gray; background-color: white">
        <div style="
          height:{(self.size + 2) * 3 + 50}px;
          width:{180}px;
          overflow-y: scroll;
          display:inline-block;
          vertical-align:top;
          overflow=y: scroll;
          'background-color': '#7b7b7b'"
          class="sidebar"><!-- sidebar -->
        <center>
        {h(self.reset_btn)}
        {h(self.sidebar)}
        </center>
        </div><!--left-->
        <!--right-->
        <div style="height:{(self.size + 2) * 3 + 50}px; width:{(self.size + 2) * 3 + 22 + 32}px;
          display: inline-block;
          vertical-align: top;
          border-left: 4px dashed gray;
          padding-left: 5px;
          margin-left: 5px;
          margin-top:8px; overflow-y: scroll">
        {show.html([[c] for c in self.image_array])}
        </div>
        </div>
        </div>'''
