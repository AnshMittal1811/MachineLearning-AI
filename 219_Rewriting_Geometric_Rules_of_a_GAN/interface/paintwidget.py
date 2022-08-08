from .show import pil_to_url
from .labwidget import Widget, Gamut, Range, Textbox, Property


class PaintWidget(Widget):
    def __init__(self,
                 width=256, height=256,
                 image='', mask='', brushsize=10.0, stroke_color=None, oneshot=False, disabled=False):
        super().__init__()
        self.mask = Property(mask)
        self.image = Property(image)
        self.brushsize = Property(brushsize)
        self.stroke_color = Property(stroke_color)
        self.erase = Property(False)
        self.oneshot = Property(oneshot)
        self.disabled = Property(disabled)
        self.width = Property(width)
        self.height = Property(height)

    def widget_js(self):
        return f'''
      {PAINT_WIDGET_JS}
      var pw = new PaintWidget(element, model);
    '''

    def widget_html(self):
        v = self.view_id()
        return f'''
    <style>
    #{v} {{ position: relative; display: inline-block; }}
    #{v} .paintmask {{
      position: absolute; top:0; left: 0; z-index: 1;
      opacity: 0.4; transition: opacity .1s ease-in-out; }}
    #{v} .paintmask:hover {{ opacity: 1.0; }}
    </style>
    <div id="{v}"></div>
    <div>
      <button id="undo_button" type="button">undo</button>
      <button id="clear_button" type="button">reset</button>
    </div>
    '''


PAINT_WIDGET_JS = """
class PaintWidget {
  constructor(el, model) {
    this.el = el;
    this.model = model;

    // canvas states
    this.cPushArray = new Array();
    this.cStep = -1;

    this.size_changed();
    this.model.on('mask', this.mask_changed.bind(this));
    this.model.on('image', this.image_changed.bind(this));
    this.model.on('width', this.size_changed.bind(this));
    this.model.on('height', this.size_changed.bind(this));
  }
  mouse_stroke(first_event) {
    var self = this;
    if (self.model.get('disabled')) { return; }
    if (self.model.get('oneshot')) {
        var canvas = self.mask_canvas;
        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    function track_mouse(evt) {
      if (evt.type == 'keydown' || self.model.get('disabled')) {
        if (self.model.get('disabled') || evt.key === "Escape") {
          window.removeEventListener('mousemove', track_mouse);
          window.removeEventListener('mouseup', track_mouse);
          window.removeEventListener('keydown', track_mouse, true);
          self.mask_changed();
        }
        return;
      }
      if (evt.type == 'mouseup' ||
        (typeof evt.buttons != 'undefined' && evt.buttons == 0)) {
        window.removeEventListener('mousemove', track_mouse);
        window.removeEventListener('mouseup', track_mouse);
        window.removeEventListener('keydown', track_mouse, true);

        // canvas states
        self.cPush();

        self.model.set('mask', self.mask_canvas.toDataURL());
        return;
      }
      var p = self.cursor_position();
      self.fill_circle(p.x, p.y,
          self.model.get('brushsize'),
          self.model.get('erase'));
    }
    this.mask_canvas.focus();
    window.addEventListener('mousemove', track_mouse);
    window.addEventListener('mouseup', track_mouse);
    window.addEventListener('keydown', track_mouse, true);
    track_mouse(first_event);
  }

  // canvas states
  undo_step(event) {
    if (this.cStep > 0) {
      this.cUndo();
      this.model.set('mask', this.cPushArray[this.cStep]);
    }
  }
  clear_canvas() {
    var canvas = this.mask_canvas;
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    this.model.set('mask', this.mask_canvas.toDataURL());
    this.cStep = 0;
  }

  mask_changed(val) {
    this.draw_data_url(this.mask_canvas, this.model.get('mask'));
  }
  image_changed() {
    this.draw_data_url(this.image_canvas, this.model.get('image'));
  }
  size_changed() {
    this.mask_canvas = document.createElement('canvas');
    this.image_canvas = document.createElement('canvas');
    this.mask_canvas.className = "paintmask";
    this.image_canvas.className = "paintimage";
    for (var attr of ['width', 'height']) {
      this.mask_canvas[attr] = this.model.get(attr);
      this.image_canvas[attr] = this.model.get(attr);
    }

    // remove anti-aliasing
    var ctx = this.mask_canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;

    this.el.innerHTML = '';
    this.el.appendChild(this.image_canvas);
    this.el.appendChild(this.mask_canvas);
    this.mask_canvas.addEventListener('mousedown',
        this.mouse_stroke.bind(this));
    this.mask_changed();
    this.image_changed();

    // canvas states
    this.cPush();
    this.undo_button = document.getElementById('undo_button');
    this.undo_button.addEventListener('click',
        this.undo_step.bind(this));

    this.clear_button = document.getElementById('clear_button');
    this.clear_button.addEventListener('click',
        this.clear_canvas.bind(this));
  }

  cursor_position(evt) {
    const rect = this.mask_canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    return {x: x, y: y};
  }

  fill_circle(x, y, r, erase, blur) {
    var ctx = this.mask_canvas.getContext('2d');
    ctx.save();
    if (blur) {
        ctx.filter = 'blur(' + blur + 'px)';
    }
    ctx.globalCompositeOperation = (
        erase ? "destination-out" : 'source-over');

    var color = model.get('stroke_color');
    if (color == null) {
      ctx.fillStyle = "gray";
    } else {
      ctx.fillStyle = color;
    }
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.restore()
  }

  draw_data_url(canvas, durl) {
    var ctx = canvas.getContext('2d');
    var img = new Image;
    canvas.pendingImg = img;
    function imgdone() {
      if (canvas.pendingImg == img) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        canvas.pendingImg = null;
      }
    }
    img.addEventListener('load', imgdone);
    img.addEventListener('error', imgdone);
    img.src = durl;
  }

  // canvas states
  cPush() {
      this.cStep++;
      if (this.cStep < this.cPushArray.length) { this.cPushArray.length = this.cStep; }
      this.cPushArray.push(this.mask_canvas.toDataURL());
  }

  cUndo() {
      if (this.cStep > 0) {
          this.cStep--;
          this.draw_data_url(this.mask_canvas, this.cPushArray[this.cStep]);
      }
  }
}
"""


class ColorPaintWidget(Widget):
    def __init__(self,
                 width=256, height=256,
                 image='', mask='', brushsize=10.0, stroke_color=None, oneshot=False, disabled=False):
        super().__init__()
        s_color = Property(stroke_color)
        b_size = Property(brushsize)
        self.gamut = Gamut(s_color)
        self.brushtext = Textbox(b_size)
        self.brushrange = Range(value=b_size, min=1, max=100)
        self.paint = PaintWidget(width, height, image, mask, b_size, s_color, oneshot, disabled)
        self.stroke_color = s_color
        self.brushsize = b_size

    def change_palette(self, palette):
        self.gamut.change_palette(palette)

    def set_image(self, image):
        self.paint.image = pil_to_url(image)

    def widget_html(self):
        def h(w):
            return w._repr_html_()
        return f'''
      <div>
        {h(self.gamut)}
        <div style="display:inline-block;vertical-align:120%;">Brush Size:</div>
        <div style="display:inline-block;vertical-align:120%;">{h(self.brushtext)}</div>
        <div style="display:inline-block;vertical-align:120%;">{h(self.brushrange)}</div>
      </div>
      <div>{h(self.paint)}</div>
    '''
