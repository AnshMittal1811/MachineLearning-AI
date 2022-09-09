from .show import pil_to_url
from .labwidget import Widget, Property


class WarpWidget(Widget):
    def __init__(self,
                 width=256, height=256,
                 image=None, keypt_init=[], brushsize=5.0, oneshot=False, disabled=False):
        super().__init__()
        self.keypt = Property(keypt_init)
        self.keypt_init = Property(keypt_init.copy())
        # self.image = Property(image)
        # self.image_warped = Property(image)
        self.brushsize = Property(brushsize)
        # self.erase = Property(False)
        self.oneshot = Property(oneshot)
        self.disabled = Property(disabled)
        self.width = Property(width)
        self.height = Property(height)

        self.source = None
        self.image = Property('')
        self.image_warped = Property('')
        if image is not None:
            self.set_new_image(image)

    def set_new_image(self, image):
        self.source = image
        self.image = pil_to_url(image)
        self.image_warped = pil_to_url(image)

    def widget_js(self):
        return f'''
      {PAINT_WIDGET_JS}
      var pw = new WarpWidget(element, model);
    '''

    def widget_html(self):
        v = self.view_id()
        return f'''
    <style>
    #{v} {{ position: relative; display: inline-block; }}
    #{v} .paintkeypt {{
      position: absolute; top:0; left: 0; z-index: 1;
      opacity: 1; transition: opacity .1s ease-in-out; }}
    //#warpimage {{ float: left; }}
    </style>
    <table><tr>
    <td><div id="{v}"></div></td>
    <td><canvas id="warpimage"></canvas></td>
    </tr></table>
    <button id="undo_button" type="button">undo</button>
    <button id="clear_button" type="button">reset</button>
    '''


PAINT_WIDGET_JS = """
class WarpWidget {
  constructor(el, model) {
    this.el = el;
    this.model = model;

    // canvas states
    this.cPushArray = new Array();
    this.cStep = -1;

    this.size_changed();
    this.model.on('keypt', this.keypt_changed.bind(this));
    this.model.on('image', this.image_changed.bind(this));
    this.model.on('image_warped', this.warp_changed.bind(this));
    this.model.on('width', this.size_changed.bind(this));
    this.model.on('height', this.size_changed.bind(this));
  }
  mouse_stroke(first_event) {
    // middle cursor selects a fixed keypoint
    if (first_event.which === 2 || first_event.button === 1) {
      this.fixed_point();
      return;
    }

    var self = this;
    if (self.model.get('disabled')) { return; }
    if (self.model.get('oneshot')) {
      clear_canvas();
    }
    function track_mouse(evt) {
      if (evt.type == 'keydown' || self.model.get('disabled')) {
        if (self.model.get('disabled') || evt.key === "Escape") {
          window.removeEventListener('mousemove', track_mouse);
          window.removeEventListener('mouseup', track_mouse);
          window.removeEventListener('keydown', track_mouse, true);
          self.keypt_changed();
        }
        return;
      }
      if (evt.type == 'mouseup' ||
        (typeof evt.buttons != 'undefined' && evt.buttons == 0)) {
        window.removeEventListener('mousemove', track_mouse);
        window.removeEventListener('mouseup', track_mouse);
        window.removeEventListener('keydown', track_mouse, true);

        var ep = self.cursor_position();
        self.fill_circle(ep.x, ep.y,
            self.model.get('brushsize'),
            '#ff0000');
        self.draw_arrow(sp.x, sp.y, ep.x, ep.y);
        self.cPush();

        self.keypt.push([sp.x, sp.y, ep.x, ep.y]);
        self.model.set('keypt', self.keypt);
        return;
      }

    }
    this.keypt_canvas.focus();
    window.addEventListener('mousemove', track_mouse);
    window.addEventListener('mouseup', track_mouse);
    window.addEventListener('keydown', track_mouse, true);

    var sp = self.cursor_position();
      self.fill_circle(sp.x, sp.y,
          self.model.get('brushsize'),
          '#00ff00');

    track_mouse(first_event);
  }

  fixed_point() {
    var self = this;
    if (self.model.get('disabled')) { return; }
    if (self.model.get('oneshot')) {
      clear_canvas();
    }

    this.keypt_canvas.focus();

    var sp = self.cursor_position();
    var cx = this.keypt_canvas.width;
    var cy = this.keypt_canvas.height;

    if (sp.x >= 0 && sp.x <= cx && sp.y >= 0 && sp.y <= cy) {
      self.fill_circle(sp.x, sp.y,
          self.model.get('brushsize'),
          '#ffff00');
      self.cPush();
      self.keypt.push([sp.x, sp.y, sp.x, sp.y]);
      // self.model.set('keypt', self.keypt); // lazy operation
    }

    return;
  }

  undo_step(event) {
    if (this.cStep > 0) {
      this.cUndo();
      this.keypt.pop();
      this.model.set('keypt', this.keypt);
    }
  }
  reset_warps(event) {
    this.clear_canvas();
    this.keypt_reset();
  }
  clear_canvas() {
    var canvas = this.keypt_canvas;
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    this.cStep = 0;
  }
  keypt_reset() {
    this.keypt = this.model.get('keypt_init');
    this.model.set('keypt', this.keypt);
  }
  keypt_changed(val) {
    this.keypt = this.model.get('keypt');
  }
  image_changed() {
    this.draw_data_url(this.image_canvas, this.model.get('image'));
  }
  warp_changed() {
    this.draw_data_url(this.image_warped, this.model.get('image_warped'));
    //this.image_warped.src = this.model.get('image_warped');
  }
  size_changed() {
    this.keypt_canvas = document.createElement('canvas');
    this.image_canvas = document.createElement('canvas');
    this.image_warped = document.getElementById('warpimage');
    this.keypt_canvas.className = "paintkeypt";
    this.image_canvas.className = "paintimage";
    this.keypt_canvas.id = "paintkeypt";
    this.image_canvas.id = "paintimage";
    // this.image_warped.className = "warpimage";
    for (var attr of ['width', 'height']) {
      this.keypt_canvas[attr] = this.model.get(attr);
      this.image_canvas[attr] = this.model.get(attr);
      this.image_warped[attr] = this.model.get(attr);
    }

    this.el.innerHTML = '';
    this.el.appendChild(this.image_canvas);
    this.el.appendChild(this.keypt_canvas);
    // this.el.appendChild(this.image_warped);
    this.keypt_canvas.addEventListener('mousedown',
        this.mouse_stroke.bind(this));
    this.keypt_changed();
    this.image_changed();
    this.warp_changed();

    this.cPush();
    this.undo_button = document.getElementById('undo_button');
    this.undo_button.addEventListener('click',
        this.undo_step.bind(this));

    this.clear_button = document.getElementById('clear_button');
    this.clear_button.addEventListener('click',
        this.reset_warps.bind(this));
  }

  cursor_position(evt) {
    const rect = this.keypt_canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    return {x: x, y: y};
  }

  fill_circle(x, y, r, color) {
    var ctx = this.keypt_canvas.getContext('2d');
    ctx.save();
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill();
    ctx.restore();
  }

  draw_arrow(fromx, fromy, tox, toy) {
    var ctx = this.keypt_canvas.getContext('2d');
    ctx.save();
    var headlen = 10; // length of head in pixels
    var dx = tox - fromx;
    var dy = toy - fromy;
    var angle = Math.atan2(dy, dx);

    ctx.beginPath();
    ctx.moveTo(fromx, fromy);
    ctx.lineTo(tox, toy);
    ctx.lineTo(tox - headlen * Math.cos(angle - Math.PI / 6), toy - headlen * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(tox, toy);
    ctx.lineTo(tox - headlen * Math.cos(angle + Math.PI / 6), toy - headlen * Math.sin(angle + Math.PI / 6));
    ctx.stroke();
    ctx.restore();
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

  cPush() {
      this.cStep++;
      if (this.cStep < this.cPushArray.length) { this.cPushArray.length = this.cStep; }
      this.cPushArray.push(document.getElementById('paintkeypt').toDataURL());
  }

  cUndo() {
      if (this.cStep > 0) {
          this.cStep--;
          this.draw_data_url(this.keypt_canvas, this.cPushArray[this.cStep]);
      }
  }
}
"""
