/* Cotton Leaf Check - core logic (model + severity). Works in browser and Node. */
(function (root) {
  const CLASSES = ['Aphids', 'Army worm', 'Bacterial Blight', 'Healthy', 'Powdery Mildew', 'Target spot'];
  const SIZE = 224;

  // Same as Keras load_img(..., interpolation='nearest') used in training. rgba: Uint8 array of w*h*4.
  function resizeNearest(rgba, w, h) {
    const out = new Uint8Array(SIZE * SIZE * 3);
    for (let y = 0; y < SIZE; y++) {
      const sy = Math.min(h - 1, Math.floor((y + 0.5) * (h / SIZE)));
      for (let x = 0; x < SIZE; x++) {
        const sx = Math.min(w - 1, Math.floor((x + 0.5) * (w / SIZE)));
        const s = (sy * w + sx) * 4, d = (y * SIZE + x) * 3;
        out[d] = rgba[s]; out[d + 1] = rgba[s + 1]; out[d + 2] = rgba[s + 2];
      }
    }
    return out;
  }

  // OpenCV-style 8-bit HSV: H 0..179, S,V 0..255
  function hsv(r, g, b) {
    const v = Math.max(r, g, b), d = v - Math.min(r, g, b);
    const s = v === 0 ? 0 : Math.round(255 * d / v);
    let h = 0;
    if (d > 0) {
      if (v === r) h = 60 * (g - b) / d; else if (v === g) h = 120 + 60 * (b - r) / d; else h = 240 + 60 * (r - g) / d;
      if (h < 0) h += 360;
    }
    return [Math.round(h / 2) % 180, s, v];
  }

  // 5x5 rectangular morphology, out-of-image pixels ignored (OpenCV default border behaviour)
  function morph(src, isMax) {
    const pass = (a, horiz) => {
      const o = new Uint8Array(a.length);
      for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
        let m = isMax ? 0 : 255;
        for (let k = -2; k <= 2; k++) {
          const xx = horiz ? x + k : x, yy = horiz ? y : y + k;
          if (xx < 0 || yy < 0 || xx >= SIZE || yy >= SIZE) continue;
          const v = a[yy * SIZE + xx];
          if (isMax ? v > m : v < m) m = v;
        }
        o[y * SIZE + x] = m;
      }
      return o;
    };
    return pass(pass(src, true), false);
  }

  // Port of calculate_infected_area() from the notebook. rgb: Uint8 224*224*3
  function infectedArea(rgb) {
    const n = SIZE * SIZE;
    let leaf = new Uint8Array(n), healthy = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const [h, s, v] = hsv(rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);
      const isLeaf = (h >= 10 && h <= 100 && s >= 20 && v >= 20) || (s <= 60 && v >= 150);
      leaf[i] = isLeaf ? 255 : 0;
      healthy[i] = (h >= 35 && h <= 85 && s >= 40 && v >= 40) ? 255 : 0;
    }
    leaf = morph(morph(leaf, true), false);   // close = dilate, erode
    leaf = morph(morph(leaf, false), true);   // open  = erode, dilate
    const mask = new Uint8Array(n);
    let leafPx = 0, infPx = 0;
    for (let i = 0; i < n; i++) {
      if (leaf[i]) {
        leafPx++;
        if (!healthy[i]) { mask[i] = 1; infPx++; }
      }
    }
    return { mask, pct: leafPx > 0 ? infPx / leafPx * 100 : 0 };
  }

  function severityLevel(pct, cls) {
    if (cls === 'Healthy') return 0;
    return pct < 20 ? 1 : pct < 45 ? 2 : 3;
  }

  // Build the network with TF.js ops from the exported (BatchNorm-folded) weights.
  function buildModel(tf, meta, buffer) {
    const W = {};
    meta.forEach(m => {
      const size = m.shape.reduce((a, b) => a * b, 1);
      W[m.name] = tf.tensor(new Float32Array(buffer, m.offset, size), m.shape);
    });
    const conv = (x, i, stride) => tf.leakyRelu(tf.add(tf.conv2d(x, W['c' + i + '_w'], stride, 'same'), W['c' + i + '_b']), 0.3);
    return function predict(input) { // input: [1,224,224,3] scaled 0..1
      return tf.tidy(() => {
        let x = conv(input, 0, 2);
        x = tf.maxPool(conv(x, 1, 1), 2, 2, 'valid');
        const g = tf.conv2d(x, W.att_W, 1, 'same');
        x = tf.mul(x, tf.sigmoid(tf.conv2d(g, W.att_f, 1, 'same')));
        x = conv(x, 2, 1).mean([1, 2]);
        x = tf.relu(tf.add(tf.matMul(x, W.d0_w), W.d0_b));
        return tf.softmax(tf.add(tf.matMul(x, W.d1_w), W.d1_b)).dataSync();
      });
    };
  }

  const api = { CLASSES, SIZE, resizeNearest, infectedArea, severityLevel, buildModel };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.CottonCore = api;
})(typeof self !== 'undefined' ? self : this);
