import os.path
import time
from collections import defaultdict

import ipywidgets
import torch
from torchvision.datasets.utils import download_url

from .distributed import primary, synchronize

__all__ = ['download', 'download_mean_latent', 'download_model', 'download_cherrypicked', 'DraggableBlobMap']


def get_model_name(model_str):
    model = 'bedrooms'
    if model_str.split(' ')[1].startswith('kitchen'):
        model = 'kitchen_living_dining'
    elif model_str.split(' ')[1].startswith('conference'):
        model = 'conference_rooms'
    return model


def download_model(model_str, path='pretrained'):
    return download(model_str, suffix='.ckpt', path=path, load=False)


def download_mean_latent(model_str, path='pretrained'):
    return download(model_str, suffix='_mean_latent.pt', path=path, load=True)


def download_cherrypicked(model_str, path='pretrained'):
    return download(model_str, suffix='_cherrypicked.pt', path=path, load=True)


def download(model_str, suffix, path, load):
    file = get_model_name(model_str) + suffix
    local_path = os.path.join(path, file)
    if not os.path.isfile(local_path) and primary():
        dl_path = f'http://efrosgans.eecs.berkeley.edu/blobgan/{file}'
        download_url(dl_path, path)
    synchronize()
    if load:
        return torch.load(local_path, map_location=lambda storage, loc: storage)
    else:
        return local_path


def clone_layout(l):
    return {k: (v if isinstance(v, bool) else (
        v.clone() if torch.is_tensor(v) else {kk: vv.clone().repeat_interleave(1, 0) for kk, vv in v.items()})) for k, v
            in l.items() if v is not None}


# Control for blob manipulation
def DraggableBlobMap(notebook_locals):
    globals().update(notebook_locals)

    # Set up display
    plt.close('all')
    plt.rcParams["figure.figsize"] = (8, 4)
    plt.ioff()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    ax0, ax1 = axes.flatten()
    plt.ion()
    ax0.axis('off')
    ax1.axis('off')
    blob_fig = ax1.imshow(labeled_blobs_img)
    img_fig = ax0.imshow(for_canvas(orig_img))
    plt.tight_layout(pad=0.5)
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.canvas.toolbar_visible = False
    out = widgets.Output()
    log = widgets.Output()

    class DraggableBlobMapClass:
        def __init__(self):
            self.handlers = {}
            self.size_threshold = size_threshold
            self.press = False
            self.L = layout
            self.L_orig = clone_layout(layout)
            self.closest = None
            self.radius = 0.02
            self.event_log = []
            self.blob_fig = blob_fig
            self.img_fig = img_fig
            self.imgs = {
                'img': for_canvas(orig_img),
                'blobs': orig_blobs,
                'labeled_blobs': labeled_blobs_img
            }
            self.orig_imgs = {
                'img': for_canvas(orig_img),
                'blobs': orig_blobs,
                'labeled_blobs': labeled_blobs_img
            }
            self.image_name = 'BlobGAN'
            self.vid_duration = 2

        def connect(self):
            self.handlers['press'] = fig.canvas.mpl_connect('button_press_event', self.onpress)
            self.handlers['release'] = fig.canvas.mpl_connect('button_release_event', self.onrelease)
            self.handlers['move'] = fig.canvas.mpl_connect('motion_notify_event', self.onmove)
            self.handlers['keypress'] = fig.canvas.mpl_connect('key_press_event', self.btn_kb_click)

        def render(self, to_canvas=True):
            # Resplat
            del self.L['feature_img']
            self.L, img = model.gen(layout=self.L, covs_raw=False, **render_kwargs)

            # Render images on canvas
            blobs = for_canvas(self.L['feature_img'].mul(255))
            labeled_blobs, labeled_blobs_img = draw_labels(blobs, self.L, self.size_threshold)
            img = for_canvas(img)
            imgdict = {
                    'img': img,
                    'blobs': blobs,
                    'labeled_blobs': labeled_blobs_img
                }

            if to_canvas:
                self.imgs = imgdict
                img_fig.set_data(img)
                blob_fig.set_data(labeled_blobs_img)
                fig.canvas.draw_idle
            else:
                return imgdict

        @out.capture()
        def update_closest(self, closest):
            color = [','.join(map(str, COLORS[c + 1].mul(255).round().int().tolist())) for c in closest]
            out.clear_output()
            blobstr = " " if len(closest) == 1 else "s "
            blobstr += ', '.join(
                [f'<span style="color:rgb({color});">{blob}</span>' for color, blob in zip(color, closest)])
            display(HTML(f'<h2 style="margin: 4px 0;">Selected blob{blobstr}</h2>'))

            self.closest = closest
            self.color = color

        @out.capture()
        def onpress(self, event):
            # Find nearest blob to click, above size threshold
            x, y = event.xdata / 256, event.ydata / 256
            mask = self.L['sizes'][0, 1:] > self.size_threshold
            idmap = torch.arange(len(mask))[mask]
            ds = (self.L['xs'][0][mask] - x) ** 2 + (self.L['ys'][0][mask] - y) ** 2
            closest = idmap[(ds - ds.min() <= self.radius).nonzero().view(-1)].tolist()

            # Update if double click or right click
            render = True
            self.press = False
            if event.button == 3:
                self.L['sizes'][0, [i + 1 for i in closest]] -= 0.25
                self.event_log.append({'t': time.perf_counter(), 'closest': closest, 'size': -0.25})
            elif event.dblclick or event.button == 2:
                self.L['sizes'][0, [i + 1 for i in closest]] += 0.25
                self.event_log.append({'t': time.perf_counter(), 'closest': closest, 'size': +0.25})
            else:
                render = False
                self.press = True

            # Print out
            self.update_closest(closest)

            # Set state
            self.xy = (x, y)
            if render: self.render()

        @out.capture()
        def threshold_slider_update(self, change):
            if change['name'] == 'value':
                # Update image with new labels
                blob_fig.set_data(draw_labels(self.imgs['blobs'], self.L, change['new'])[1])
                self.size_threshold = change['new']
                fig.canvas.draw_idle

        @out.capture()
        def selector_update(self, change):
            if change['name'] == 'value':
                new = list(map(int, [blob for blob in change['new'].split(',') if blob.isnumeric()]))
                if len(new): self.update_closest(new)

        @out.capture()
        def name_text_update(self, change):
            if change['name'] == 'value':
                new = change['new']
                if new:
                    self.image_name = new

        @out.capture()
        def onmove(self, event):
            # Mouse down?
            if not self.press: return

            # Update layout
            pressx, pressy = self.xy
            x, y = event.xdata / 256, event.ydata / 256
            self.event_log.append(
                {'t': time.perf_counter(), 'closest': self.closest, 'x': x, 'y': y, 'dx': x - pressx, 'dy': y - pressy})
            self.L['xs'][0, self.closest] += x - pressx
            self.L['ys'][0, self.closest] += y - pressy
            self.xy = (x, y)

            # Re-splat blobs and render new image
            self.render()

        @out.capture()
        def onrelease(self, event):
            self.press = False

        # @log.capture()
        def save_btn_click(self, event):
            btn = event.description.lower()
            log.clear_output()
            if 'video' in btn:
                imgs, blobs = [self.imgs['img']], [self.imgs['blobs']]
                n_events = len(self.event_log)
                self.L = clone_layout(self.L_orig)
                with log:
                    for e in tqdm(self.event_log, desc='Generating video'):
                        ids = e['closest']
                        n_frames = round((self.vid_duration * record_fps) / n_events)
                        for _ in range(n_frames):
                            if 'dx' in e:
                                self.L['xs'][0, ids] += e['dx'] / n_frames
                            if 'dy' in e:
                                self.L['ys'][0, ids] += e['dy'] / n_frames
                            if 'size' in e:
                                self.L['sizes'][0, [i+1 for i in ids]] += e['size'] / n_frames
                            out = self.render(to_canvas=False)
                            imgs.append(out['img'])
                            blobs.append(out['blobs'])
                img_clip = concatenate_videoclips([ImageClip(i).set_duration(1/record_fps).set_fps(record_fps) for i in imgs], method="compose")
                img_clip_name = f'{self.image_name}_image.mp4'
                img_clip.set_fps(record_fps).write_videofile(img_clip_name, verbose=False, logger=None)
                blob_clip = concatenate_videoclips([ImageClip(i).set_duration(1/record_fps).set_fps(record_fps) for i in blobs], method="compose")
                blob_clip_name = f'{self.image_name}_blobs.mp4'
                blob_clip.set_fps(record_fps).write_videofile(blob_clip_name, verbose=False, logger=None)
                with log:
                    print(
                        '\n \033[95m If prompted, make sure to allow your browser to download multiple files on this webpage.')
                files.download(img_clip_name)
                files.download(blob_clip_name)
                self.L = clone_layout(self.L_orig)

            else:
                if 'image' in btn:
                    name = f'{self.image_name}_image.png'
                    img = Image.fromarray(self.imgs['img'])
                elif 'labeled' in btn:
                    img = self.imgs['labeled_blobs']
                    name = f'{self.image_name}_labeled_blobs.png'
                elif 'blobs' in btn:
                    img = Image.fromarray(self.imgs['blobs'])
                    name = f'{self.image_name}_blobs.png'
                img.save(name)
                files.download(name)

        def vid_dur_update(self, change):
            if change['name'] == 'value':
                new = change['new']
                self.vid_duration = new

        @out.capture()
        def btn_kb_click(self, event):
            if self.closest is None:
                out.clear_output()
                display(
                    HTML(f'<h2 style="margin: 4px 0;">Select a blob before clicking buttons or pressing keys!</h2>'))
                return
            try:
                desc = event.description.lower()  # Button click
            except:
                desc = event.key.lower()  # Key press
            event_info = {}
            if desc == 'shrink' or desc == '-' or desc == '_':
                self.L['sizes'][0, [i + 1 for i in self.closest]] -= 0.25
                event_info['size'] = -0.25
            elif desc == 'enlarge' or desc == '+' or desc == '=':
                self.L['sizes'][0, [i + 1 for i in self.closest]] += 0.25
                event_info['size'] = 0.25
            elif desc == 'left':
                self.L['xs'][0, self.closest] -= 0.05
                event_info['dx'] = -0.05
            elif desc == 'right':
                self.L['xs'][0, self.closest] += 0.05
                event_info['dx'] = 0.05
            elif desc == 'up':
                self.L['ys'][0, self.closest] -= 0.05
                event_info['dy'] = -0.05
            elif desc == 'down':
                self.L['ys'][0, self.closest] += 0.05
                event_info['dy'] = 0.05
            elif desc.startswith('reset') or desc == 'r':
                self.L = clone_layout(self.L_orig)
                event_info['reset'] = True
            self.event_log.append({'t': time.perf_counter(), 'closest': self.closest, **event_info})
            if event_info.get('reset', False):
                self.event_log = []
            self.render()

    # Set up UI layout (not super beautiful, but it works)
    blob_control = DraggableBlobMapClass()
    blob_control.connect()

    blob_filter_slider = widgets.BoundedFloatText(value=blob_control.size_threshold,
                                                  min=-5, max=5, step=0.1,
                                                  description='View blobs above size:',
                                                  layout=widgets.Layout(margin='0 0 0 10px', width='200px'),
                                                  style={'description_width': 'initial'})
    blob_filter_slider.observe(blob_control.threshold_slider_update, names='value')

    width_auto = widgets.Layout(width='auto')
    btn_specific_layout = defaultdict(lambda: dict(width='100px'))
    btn_specific_layout['Reset scene'] = dict(margin='0', width='150px')
    buttons = [widgets.Button(
        description=desc,
        layout=widgets.Layout(grid_area=desc, **btn_specific_layout[desc]),
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip=desc,
        icon=icon  # (FontAwesome names without the `fa-` prefix)
    ) for desc, icon in zip(['Shrink', 'Enlarge', 'Left', 'Right', 'Up', 'Down', 'Reset scene'],
                            ['minus', 'plus', 'arrow-left', 'arrow-right', 'arrow-up', 'arrow-down', 'refresh'])]
    [b.on_click(blob_control.btn_kb_click) for b in buttons]

    save_buttons = [widgets.Button(
        description=desc,
        layout=widgets.Layout(width='130px' if 'video' in desc else '150px'),
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip=desc,
        icon=icon  # (FontAwesome names without the `fa-` prefix)
    ) for desc, icon in zip(['Create video', 'Save image', 'Save blobs', 'Save labeled blobs'],
                            ['video-camera', 'floppy-o', 'floppy-o', 'floppy-o'])]
    [b.on_click(blob_control.save_btn_click) for b in save_buttons]

    blob_selector = widgets.Text(
        value='',
        placeholder='e.g. "5" or "11, 6"',
        description='Select blob(s):',
        style={'description_width': 'initial'},
        disabled=False
    )
    blob_selector.observe(blob_control.selector_update, names='value')

    name_text = widgets.Text(
        value='BlobGAN',
        placeholder='BlobGAN',
        description='Image name:',
        disabled=False,
        layout=widgets.Layout(width='97%', max_width='400px')
    )
    name_text.observe(blob_control.name_text_update, names='value')

    vid_dur_slider = widgets.BoundedFloatText(value=blob_control.vid_duration,
                                              min=1, max=20, step=1,
                                              description='Duration (sec):',
                                              layout=widgets.Layout(margin='0 0 0 5px', width='140px'),
                                              style={'description_width': 'initial'})
    vid_dur_slider.observe(blob_control.vid_dur_update, names='value')

    # Render and get started!
    display(widgets.HBox([widgets.VBox([

        widgets.HBox([blob_selector, blob_filter_slider]),
        widgets.HBox([fig.canvas], layout=width_auto),
        widgets.HBox([
            widgets.VBox([name_text,
                          widgets.HBox([save_buttons[0], vid_dur_slider]),
                          *save_buttons[1:], buttons[-1]],
                         layout=widgets.Layout(width='48%', align_items='center')),
            widgets.VBox([
                widgets.HBox([out], layout=widgets.Layout(max_width='280px', padding='5px')),
                widgets.HBox([buttons[0], buttons[1]]),
                widgets.GridBox(children=buttons[2:-1], layout=widgets.Layout(
                    width='210px',
                    grid_template_columns='1fr 1fr 1fr 1fr',
                    grid_template_rows='auto auto auto',
                    grid_template_areas='''
            ". Up Up ."
            "Left Left Right Right"
            ". Down Down ."
            '''
                ))], layout=widgets.Layout(width='52%', align_items='center')),
        ])
    ])]))
    display(log)
    # display(widgets.VBox([fig.canvas, widgets.HBox([blob_filter_slider, blob_selector]), widgets.HBox(buttons)]))

    with out:
        display(HTML(f'<h2 style="margin: 4px 0;">Click and drag an object in the image!</h2>'))

    return blob_control
