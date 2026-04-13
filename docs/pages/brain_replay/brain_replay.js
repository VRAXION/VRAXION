/*
 * Vraxion Brain Replay — vanilla JS replay visualizer.
 *
 * No frameworks. No CDN. No ES modules. Plain IIFE exposing window.BrainReplay.
 *
 * Consumers:
 *   index.html  -> window.BrainReplay.fetchTasks()
 *   replay.html -> window.BrainReplay.init()
 *
 * Trace JSON schema (frozen by v5_grid3_design_spec.md §"Trace JSON schema"):
 *   {
 *     task, data_seed, search_seed, n_in, n_per, noise,
 *     events: [{event,tick,id,parents,weights,threshold,alpha,train_acc,val_acc}, ...],
 *     final: {best_val_acc, best_test_acc, total_neurons, max_depth, stall_count}
 *   }
 *
 * tasks.json schema (frozen by Worker HARNESS section):
 *   { generated_utc, tasks: [{task,best_val_acc,total_neurons,max_depth,winner_seed}, ...] }
 */
(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';

  // ---- Graph viewBox geometry (keep in sync with replay.html viewBox="0 0 900 520") ----
  var GRAPH_W = 900;
  var GRAPH_H = 520;
  var INPUT_Y = 50;
  var INPUT_COUNT = 9;
  var INPUT_R = 16;
  var NEURON_R = 14;
  // Row y positions keyed by tick (depth). tick 0 == input row.
  var ROW_Y = {
    0: INPUT_Y,
    1: 145,
    2: 230,
    3: 315,
    4: 395,
    5: 465
  };
  var ROW_Y_FALLBACK = 465; // used for any tick >= 6

  // ---- Chart viewBox geometry (replay.html viewBox="0 0 520 320") ----
  var CHART_W = 520;
  var CHART_H = 320;
  var CHART_MARGIN = { top: 28, right: 20, bottom: 42, left: 48 };
  var CHART_INNER_W = CHART_W - CHART_MARGIN.left - CHART_MARGIN.right;
  var CHART_INNER_H = CHART_H - CHART_MARGIN.top - CHART_MARGIN.bottom;

  // ---- Replay state ----
  var state = {
    trace: null,      // parsed trace.json or null
    step: 0,          // number of events rendered, 0..trace.events.length
    playing: false,
    speed: 1,
    timerId: null,
    layout: null,     // Map<id, {x,y,tick}>
    nodeEls: null,    // Map<id, SVGGElement> for hidden neurons
    edgeEls: null,    // Map<"id-parent", SVGLineElement> for edges
    chartPoly: null,  // SVGPolylineElement for val_acc trajectory
    chartDot: null,   // SVGCircleElement for current event marker
    graphSvg: null,
    chartSvg: null,
    logEl: null,
    stepCountEl: null,
    scrubEl: null,
    playBtn: null
  };

  // =====================================================================
  // INDEX PAGE — fetchTasks
  // =====================================================================
  function fetchTasks() {
    var grid = document.getElementById('task-grid');
    var headerCount = document.getElementById('header-count');
    if (!grid) return;

    fetch('./tasks.json', { cache: 'no-store' })
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(function (idx) {
        var tasks = (idx && idx.tasks) || [];
        if (!tasks.length) {
          renderEmpty('tasks.json loaded but contains 0 tasks. Run tools/run_grid3_curriculum.py.');
          return;
        }
        renderTaskCards(grid, tasks);
        if (headerCount) {
          headerCount.textContent = tasks.length + ' task' + (tasks.length === 1 ? '' : 's');
        }
      })
      .catch(function (err) {
        renderEmpty(
          'tasks.json not found yet. Wave 5 HARNESS will create it once the curriculum runs.',
          err && err.message
        );
      });
  }

  function renderTaskCards(grid, tasks) {
    grid.innerHTML = '';
    for (var i = 0; i < tasks.length; i++) {
      var t = tasks[i] || {};
      var taskName = t.task || ('task_' + i);
      var a = document.createElement('a');
      a.href = './replay.html?task=' + encodeURIComponent(taskName);
      a.className = 'card';
      a.setAttribute('role', 'link');
      a.setAttribute('aria-label', 'Replay ' + taskName);

      var h2 = document.createElement('h2');
      h2.textContent = taskName;
      a.appendChild(h2);

      a.appendChild(kv('val_acc', fmtPct(t.best_val_acc)));
      a.appendChild(kv('neurons', fmtInt(t.total_neurons)));
      a.appendChild(kv('depth', fmtInt(t.max_depth)));
      a.appendChild(kv('seed', fmtInt(t.winner_seed)));

      grid.appendChild(a);
    }
  }

  function kv(label, value) {
    var row = document.createElement('div');
    row.className = 'kv';
    var s = document.createElement('span');
    s.textContent = label;
    var b = document.createElement('b');
    b.textContent = value;
    row.appendChild(s);
    row.appendChild(b);
    return row;
  }

  function fmtPct(v) {
    if (v == null || isNaN(v)) return '—';
    return (Math.round(v * 10) / 10).toFixed(1) + '%';
  }

  function fmtInt(v) {
    if (v == null || isNaN(v)) return '—';
    return String(v | 0);
  }

  function renderEmpty(message, detail) {
    var grid = document.getElementById('task-grid');
    var slot = document.getElementById('empty-slot');
    if (grid) grid.innerHTML = '';
    if (!slot) return;
    slot.innerHTML = '';
    var box = document.createElement('div');
    box.className = 'empty-state';
    var p = document.createElement('p');
    p.textContent = message;
    box.appendChild(p);
    if (detail) {
      var pre = document.createElement('pre');
      pre.style.fontSize = '11px';
      pre.style.color = 'var(--muted)';
      pre.style.marginTop = '0.5rem';
      pre.textContent = String(detail);
      box.appendChild(pre);
    }
    slot.appendChild(box);
  }

  // =====================================================================
  // REPLAY PAGE — init / loadTrace / initReplay
  // =====================================================================
  function init() {
    state.graphSvg = document.getElementById('graph');
    state.chartSvg = document.getElementById('chart');
    state.logEl = document.getElementById('log');
    state.stepCountEl = document.getElementById('step-count');
    state.scrubEl = document.getElementById('scrub');
    state.playBtn = document.getElementById('btn-play');

    var titleEl = document.getElementById('task-title');
    var statsEl = document.getElementById('task-stats');

    var task = getQueryParam('task') || '';
    if (!task) {
      if (titleEl) titleEl.textContent = 'no task specified';
      showError('No ?task=<name> in URL. Return to the task list.');
      wireTransport(); // wire buttons so they are at least inert, not broken
      return;
    }

    if (titleEl) titleEl.textContent = task;
    if (statsEl) statsEl.textContent = 'loading trace…';

    loadTrace(task)
      .then(function (trace) {
        if (statsEl) statsEl.textContent = describeTrace(trace);
        initReplay(trace);
      })
      .catch(function (err) {
        if (statsEl) statsEl.textContent = 'trace unavailable';
        showError(
          'No trace available for "' + task + '". ' +
          'The Wave 5 HARNESS run may not have produced traces/' + task + '.json yet.',
          err && err.message
        );
        // Still render an empty skeleton so panes do not collapse.
        initReplay(makeEmptyTrace(task));
      });
  }

  function loadTrace(taskName) {
    var url = './traces/' + encodeURIComponent(taskName) + '.json';
    return fetch(url, { cache: 'no-store' }).then(function (res) {
      if (!res.ok) throw new Error('HTTP ' + res.status + ' for ' + url);
      return res.json();
    });
  }

  function makeEmptyTrace(task) {
    return {
      task: task,
      data_seed: 0,
      search_seed: 0,
      n_in: INPUT_COUNT,
      n_per: 0,
      noise: 0,
      events: [],
      final: {
        best_val_acc: 0,
        best_test_acc: 0,
        total_neurons: 0,
        max_depth: 0,
        stall_count: 0
      }
    };
  }

  function describeTrace(trace) {
    var f = (trace && trace.final) || {};
    var parts = [];
    if (f.best_val_acc != null) parts.push('val ' + fmtPct(f.best_val_acc));
    if (f.best_test_acc != null) parts.push('test ' + fmtPct(f.best_test_acc));
    if (f.total_neurons != null) parts.push(fmtInt(f.total_neurons) + ' neurons');
    if (f.max_depth != null) parts.push('depth ' + fmtInt(f.max_depth));
    if (trace && trace.search_seed != null) parts.push('seed ' + fmtInt(trace.search_seed));
    return parts.join(' · ');
  }

  function getQueryParam(name) {
    var s = (window.location.search || '').replace(/^\?/, '');
    var parts = s.split('&');
    for (var i = 0; i < parts.length; i++) {
      var eq = parts[i].indexOf('=');
      var k = eq >= 0 ? parts[i].substring(0, eq) : parts[i];
      if (k === name) {
        var v = eq >= 0 ? parts[i].substring(eq + 1) : '';
        try { return decodeURIComponent(v.replace(/\+/g, ' ')); }
        catch (e) { return v; }
      }
    }
    return null;
  }

  function showError(msg, detail) {
    var slot = document.getElementById('error-slot');
    if (!slot) return;
    slot.innerHTML = '';
    var box = document.createElement('div');
    box.className = 'error-box';
    box.textContent = msg;
    if (detail) {
      var pre = document.createElement('pre');
      pre.style.marginTop = '0.4rem';
      pre.style.fontSize = '11px';
      pre.textContent = String(detail);
      box.appendChild(pre);
    }
    slot.appendChild(box);
  }

  // ---- initReplay ----
  function initReplay(trace) {
    state.trace = trace;
    state.step = 0;
    state.playing = false;
    state.layout = layoutNeurons(trace.events || [], trace.n_in || INPUT_COUNT);
    state.nodeEls = new Map();
    state.edgeEls = new Map();

    buildGraphSvg(trace);
    buildChartSvg(trace);
    wireTransport();

    var total = (trace.events || []).length;
    if (state.scrubEl) {
      state.scrubEl.min = '0';
      state.scrubEl.max = String(total);
      state.scrubEl.value = '0';
      state.scrubEl.disabled = total === 0;
    }

    setStep(0);
  }

  // =====================================================================
  // LAYOUT — compute (x,y,tick) for every neuron
  // =====================================================================
  function layoutNeurons(events, nIn) {
    // events: array of neuron_added events. Each event has {id, parents, tick, ...}.
    // Input IDs are 0..nIn-1. Hidden neuron IDs start at nIn.
    // We do NOT trust the provided tick value alone; we compute depth = 1 + max(depth(parent))
    // so the visualizer is robust to any tick scheme. Input depth = 0.
    var layout = new Map();

    // Seed inputs row.
    for (var i = 0; i < nIn; i++) {
      layout.set(i, {
        x: inputX(i, nIn),
        y: INPUT_Y,
        tick: 0,
        isInput: true
      });
    }

    // Compute depth for each neuron event.
    var depthOf = new Map();
    for (var k = 0; k < nIn; k++) depthOf.set(k, 0);
    var rowMembers = new Map(); // depth -> array of neuron ids (hidden only)

    for (var e = 0; e < events.length; e++) {
      var ev = events[e] || {};
      if (ev.event && ev.event !== 'neuron_added') continue;
      var id = ev.id | 0;
      var globalId = nIn + id;
      var parents = Array.isArray(ev.parents) ? ev.parents : [];
      var d = 0;
      for (var p = 0; p < parents.length; p++) {
        var pid = parents[p] | 0;
        var pd = depthOf.has(pid) ? depthOf.get(pid) : 0;
        if (pd + 1 > d) d = pd + 1;
      }
      if (d < 1) d = 1;
      depthOf.set(globalId, d);
      if (!rowMembers.has(d)) rowMembers.set(d, []);
      rowMembers.get(d).push(globalId);
    }

    // Now assign x/y based on depth and order within row.
    rowMembers.forEach(function (ids, depth) {
      var y = rowY(depth);
      var n = ids.length;
      for (var i = 0; i < n; i++) {
        var id = ids[i];
        var x = neuronX(i, n);
        layout.set(id, {
          x: x,
          y: y,
          tick: depth,
          isInput: false
        });
      }
    });

    return layout;
  }

  function inputX(i, nIn) {
    // Evenly space inputs across the graph width with some padding.
    var pad = 60;
    var usable = GRAPH_W - 2 * pad;
    if (nIn <= 1) return GRAPH_W / 2;
    return pad + (usable * i) / (nIn - 1);
  }

  function neuronX(i, nRow) {
    var pad = 90;
    var usable = GRAPH_W - 2 * pad;
    if (nRow <= 1) return GRAPH_W / 2;
    return pad + (usable * i) / (nRow - 1);
  }

  function rowY(depth) {
    if (depth in ROW_Y) return ROW_Y[depth];
    return ROW_Y_FALLBACK;
  }

  // =====================================================================
  // SVG BUILD — graph
  // =====================================================================
  function buildGraphSvg(trace) {
    var svg = state.graphSvg;
    if (!svg) return;
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    var defs = el('defs');
    svg.appendChild(defs);

    // Row labels on the left margin.
    var rowDepths = collectRowDepths(trace.events || []);
    rowDepths.forEach(function (depth) {
      var t = el('text', {
        class: 'row-label',
        x: 12,
        y: rowY(depth) + 3
      });
      t.textContent = 'tick ' + depth;
      svg.appendChild(t);
    });
    // Always label the input row.
    var inputLabel = el('text', {
      class: 'row-label',
      x: 12,
      y: INPUT_Y + 3
    });
    inputLabel.textContent = 'inputs';
    svg.appendChild(inputLabel);

    // 1) Inputs (always visible).
    var nIn = trace.n_in || INPUT_COUNT;
    for (var i = 0; i < nIn; i++) {
      var ix = inputX(i, nIn);
      var g = el('g', { class: 'input-group' });
      g.appendChild(el('circle', {
        class: 'input',
        cx: ix,
        cy: INPUT_Y,
        r: INPUT_R
      }));
      var lab = el('text', {
        class: 'input-label',
        x: ix,
        y: INPUT_Y + 1
      });
      lab.textContent = 'x' + i;
      g.appendChild(lab);
      svg.appendChild(g);
    }

    // 2) Pre-create hidden neurons + edges, initially display:none.
    var events = trace.events || [];
    var edgeGroup = el('g', { class: 'edges' });
    var nodeGroup = el('g', { class: 'nodes' });
    svg.appendChild(edgeGroup);
    svg.appendChild(nodeGroup);

    for (var ev = 0; ev < events.length; ev++) {
      var e = events[ev];
      if (!e || (e.event && e.event !== 'neuron_added')) continue;
      var id = e.id | 0;
      var gid = nIn + id;
      var info = state.layout.get(gid);
      if (!info) continue;

      // Edges from each parent to this neuron.
      var parents = Array.isArray(e.parents) ? e.parents : [];
      var weights = Array.isArray(e.weights) ? e.weights : [];
      for (var p = 0; p < parents.length; p++) {
        var pid = parents[p] | 0;
        var pinfo = state.layout.get(pid);
        if (!pinfo) continue;
        var w = weights[p] != null ? (weights[p] | 0) : 0;
        var cls = 'edge ' + weightClass(w);
        var line = el('line', {
          class: cls,
          x1: pinfo.x,
          y1: pinfo.y,
          x2: info.x,
          y2: info.y
        });
        line.style.display = 'none';
        edgeGroup.appendChild(line);
        state.edgeEls.set(gid + '-' + pid + '-' + p, line);
      }

      // Neuron group.
      var g = el('g', { class: 'neuron-group', id: 'neuron-' + id });
      g.style.display = 'none';
      g.appendChild(el('circle', {
        class: 'neuron',
        cx: info.x,
        cy: info.y,
        r: NEURON_R
      }));
      var t = el('text', {
        class: 'neuron-label',
        x: info.x,
        y: info.y + 1
      });
      t.textContent = 'N' + id;
      g.appendChild(t);
      nodeGroup.appendChild(g);
      state.nodeEls.set(gid, g);
    }
  }

  function collectRowDepths(events) {
    var set = {};
    state.layout.forEach(function (info) {
      if (!info.isInput) set[info.tick] = true;
    });
    var out = [];
    Object.keys(set).forEach(function (k) { out.push(k | 0); });
    out.sort(function (a, b) { return a - b; });
    return out;
  }

  function weightClass(w) {
    if (w > 0) return 'edge-pos';
    if (w < 0) return 'edge-neg';
    return 'edge-zero';
  }

  // =====================================================================
  // SVG BUILD — chart
  // =====================================================================
  function buildChartSvg(trace) {
    var svg = state.chartSvg;
    if (!svg) return;
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    var events = trace.events || [];
    var n = events.length;

    // Plot area rectangle.
    var x0 = CHART_MARGIN.left;
    var y0 = CHART_MARGIN.top;
    var x1 = x0 + CHART_INNER_W;
    var y1 = y0 + CHART_INNER_H;

    // Gridlines at 25/50/75/100 and 0.
    [0, 25, 50, 75, 100].forEach(function (pct) {
      var y = yForPct(pct);
      svg.appendChild(el('line', {
        class: 'gridline',
        x1: x0, y1: y, x2: x1, y2: y
      }));
      var t = el('text', {
        class: 'axis-label',
        x: x0 - 6, y: y + 3,
        'text-anchor': 'end'
      });
      t.textContent = pct + '%';
      svg.appendChild(t);
    });

    // Axes.
    svg.appendChild(el('line', {
      class: 'axis',
      x1: x0, y1: y1, x2: x1, y2: y1
    }));
    svg.appendChild(el('line', {
      class: 'axis',
      x1: x0, y1: y0, x2: x0, y2: y1
    }));

    // X-axis ticks (max 6 ticks).
    var tickCount = Math.min(Math.max(n, 1), 6);
    for (var i = 0; i <= tickCount; i++) {
      var frac = tickCount === 0 ? 0 : i / tickCount;
      var x = x0 + frac * CHART_INNER_W;
      svg.appendChild(el('line', {
        class: 'axis',
        x1: x, y1: y1, x2: x, y2: y1 + 4
      }));
      var idx = Math.round(frac * Math.max(n, 1));
      var lbl = el('text', {
        class: 'axis-label',
        x: x, y: y1 + 16,
        'text-anchor': 'middle'
      });
      lbl.textContent = String(idx);
      svg.appendChild(lbl);
    }

    // X-axis caption.
    var xcap = el('text', {
      class: 'axis-label',
      x: x0 + CHART_INNER_W / 2,
      y: CHART_H - 6,
      'text-anchor': 'middle'
    });
    xcap.textContent = 'event index';
    svg.appendChild(xcap);

    // Y-axis caption.
    var ycap = el('text', {
      class: 'axis-label',
      x: 12,
      y: y0 + CHART_INNER_H / 2,
      'text-anchor': 'middle',
      transform: 'rotate(-90, 12, ' + (y0 + CHART_INNER_H / 2) + ')'
    });
    ycap.textContent = 'val_acc %';
    svg.appendChild(ycap);

    // Polyline (empty until setStep).
    var poly = el('polyline', {
      class: 'poly',
      points: ''
    });
    svg.appendChild(poly);
    state.chartPoly = poly;

    // Current-event marker (hidden initially).
    var dot = el('circle', {
      class: 'poly-dot',
      cx: -10, cy: -10, r: 4
    });
    dot.style.display = 'none';
    svg.appendChild(dot);
    state.chartDot = dot;

    // "No events" overlay when trace is empty.
    if (n === 0) {
      var msg = el('text', {
        class: 'axis-label',
        x: x0 + CHART_INNER_W / 2,
        y: y0 + CHART_INNER_H / 2,
        'text-anchor': 'middle'
      });
      msg.setAttribute('fill', 'var(--muted)');
      msg.textContent = 'no events';
      svg.appendChild(msg);
    }
  }

  function xForIdx(i, total) {
    if (total <= 1) return CHART_MARGIN.left + CHART_INNER_W / 2;
    return CHART_MARGIN.left + (CHART_INNER_W * i) / (total - 1);
  }

  function yForPct(pct) {
    return CHART_MARGIN.top + CHART_INNER_H * (1 - pct / 100);
  }

  // =====================================================================
  // TRANSPORT + STEP
  // =====================================================================
  function wireTransport() {
    var first = document.getElementById('btn-first');
    var back = document.getElementById('btn-back');
    var play = document.getElementById('btn-play');
    var fwd = document.getElementById('btn-fwd');
    var last = document.getElementById('btn-last');
    var speed = document.getElementById('speed');
    var scrub = document.getElementById('scrub');

    if (first) first.onclick = function () { pause(); setStep(0); };
    if (back) back.onclick = function () { pause(); stepBackward(); };
    if (fwd) fwd.onclick = function () { pause(); stepForward(); };
    if (last) last.onclick = function () {
      pause();
      setStep(((state.trace && state.trace.events) || []).length);
    };
    if (play) play.onclick = function () {
      if (state.playing) pause(); else playTransport();
    };
    if (speed) speed.onchange = function () {
      state.speed = parseFloat(speed.value) || 1;
      if (state.playing) {
        pause();
        playTransport();
      }
    };
    if (scrub) scrub.oninput = function () {
      pause();
      scrubTo(parseInt(scrub.value, 10) || 0);
    };
  }

  function playTransport() {
    var events = (state.trace && state.trace.events) || [];
    if (!events.length) return;
    if (state.step >= events.length) state.step = 0; // rewind if at end
    state.playing = true;
    if (state.playBtn) {
      state.playBtn.innerHTML = '&#x23F8;'; // pause glyph
      state.playBtn.classList.add('active');
    }
    var interval = Math.max(80, 600 / state.speed);
    state.timerId = window.setInterval(function () {
      if (state.step >= events.length) {
        pause();
        return;
      }
      stepForward();
    }, interval);
  }

  function pause() {
    state.playing = false;
    if (state.timerId) {
      window.clearInterval(state.timerId);
      state.timerId = null;
    }
    if (state.playBtn) {
      state.playBtn.innerHTML = '&#x25B6;';
      state.playBtn.classList.remove('active');
    }
  }

  function stepForward() {
    var events = (state.trace && state.trace.events) || [];
    if (state.step < events.length) setStep(state.step + 1);
  }

  function stepBackward() {
    if (state.step > 0) setStep(state.step - 1);
  }

  function scrubTo(k) {
    var total = ((state.trace && state.trace.events) || []).length;
    if (k < 0) k = 0;
    if (k > total) k = total;
    setStep(k);
  }

  function setStep(k) {
    var trace = state.trace;
    if (!trace) return;
    var events = trace.events || [];
    var total = events.length;
    if (k < 0) k = 0;
    if (k > total) k = total;
    state.step = k;

    // Show neurons + edges for events[0..k). Hide the rest. Mark the last-added as current.
    // Neurons.
    state.nodeEls.forEach(function (nodeEl, id) {
      nodeEl.style.display = 'none';
      var circle = nodeEl.querySelector('circle');
      if (circle) circle.setAttribute('class', 'neuron');
    });
    // Edges.
    state.edgeEls.forEach(function (line) {
      line.style.display = 'none';
      var cls = line.getAttribute('class') || '';
      cls = cls.replace(/\s*current\s*/g, ' ').trim();
      line.setAttribute('class', cls);
    });

    var nInStep = (state.trace && state.trace.n_in) || INPUT_COUNT;
    for (var i = 0; i < k; i++) {
      var ev = events[i];
      if (!ev) continue;
      var id = ev.id | 0;
      var gid = nInStep + id;
      var nodeEl = state.nodeEls.get(gid);
      if (nodeEl) {
        nodeEl.style.display = '';
        var circle = nodeEl.querySelector('circle');
        if (circle && i === k - 1) {
          circle.setAttribute('class', 'neuron current');
        }
      }
      var parents = Array.isArray(ev.parents) ? ev.parents : [];
      for (var p = 0; p < parents.length; p++) {
        var pid = parents[p] | 0;
        var line = state.edgeEls.get(gid + '-' + pid + '-' + p);
        if (line) {
          line.style.display = '';
          if (i === k - 1) {
            var cls = line.getAttribute('class') || '';
            if (cls.indexOf('current') < 0) {
              line.setAttribute('class', cls + ' current');
            }
          }
        }
      }
    }

    // Reorder: bring current neuron on top by re-appending.
    if (k > 0) {
      var curEv = events[k - 1];
      if (curEv) {
        var curNode = state.nodeEls.get(nInStep + (curEv.id | 0));
        if (curNode && curNode.parentNode) curNode.parentNode.appendChild(curNode);
      }
    }

    // Update chart polyline from events[0..k).
    updateChart(events, k);

    // Update log pane.
    renderLog(trace, k);

    // Update transport UI.
    if (state.scrubEl) state.scrubEl.value = String(k);
    if (state.stepCountEl) state.stepCountEl.textContent = k + ' / ' + total;
  }

  function updateChart(events, k) {
    if (!state.chartPoly) return;
    var total = events.length;
    if (total === 0) {
      state.chartPoly.setAttribute('points', '');
      if (state.chartDot) state.chartDot.style.display = 'none';
      return;
    }
    var pts = [];
    for (var i = 0; i < k; i++) {
      var v = events[i] && events[i].val_acc != null ? Number(events[i].val_acc) : 0;
      if (isNaN(v)) v = 0;
      var x = xForIdx(i, total);
      var y = yForPct(clamp(v, 0, 100));
      pts.push(x.toFixed(2) + ',' + y.toFixed(2));
    }
    state.chartPoly.setAttribute('points', pts.join(' '));
    if (state.chartDot) {
      if (k > 0) {
        var last = events[k - 1];
        var lv = last && last.val_acc != null ? Number(last.val_acc) : 0;
        if (isNaN(lv)) lv = 0;
        state.chartDot.setAttribute('cx', xForIdx(k - 1, total).toFixed(2));
        state.chartDot.setAttribute('cy', yForPct(clamp(lv, 0, 100)).toFixed(2));
        state.chartDot.style.display = '';
      } else {
        state.chartDot.style.display = 'none';
      }
    }
  }

  function renderLog(trace, k) {
    var el = state.logEl;
    if (!el) return;
    var events = trace.events || [];
    if (!events.length) {
      el.innerHTML = '<span class="log-empty">no events in trace</span>';
      return;
    }
    var lines = [];
    var nIn = trace.n_in || INPUT_COUNT;
    for (var i = 0; i < k; i++) {
      lines.push(formatEvent(events[i], nIn, i));
    }
    if (!lines.length) {
      el.innerHTML = '<span class="log-empty">press play or step forward &#x23E9;</span>';
      return;
    }
    el.textContent = lines.join('\n');
    // Auto-scroll to bottom so the newest event is visible.
    el.scrollTop = el.scrollHeight;
  }

  function formatEvent(ev, nIn, idx) {
    if (!ev) return '#' + pad(idx, 3) + '  (null event)';
    var parents = Array.isArray(ev.parents) ? ev.parents : [];
    var weights = Array.isArray(ev.weights) ? ev.weights : [];
    var parentNames = [];
    var wStr = '';
    for (var i = 0; i < parents.length; i++) {
      var pid = parents[i] | 0;
      parentNames.push(pid < nIn ? ('x' + pid) : ('N' + (pid - nIn)));
      var w = weights[i] != null ? (weights[i] | 0) : 0;
      wStr += (w > 0 ? '+' : (w < 0 ? '-' : '0'));
    }
    return (
      '#' + pad(idx, 3) +
      '  tick=' + pad(ev.tick, 3) +
      '  N' + pad(ev.id, 3) +
      '  parents=[' + parentNames.join(',') + ']' +
      '  w=' + wStr +
      '  thr=' + (ev.threshold != null ? ev.threshold : '?') +
      '  alpha=' + (ev.alpha != null ? Number(ev.alpha).toFixed(3) : '?') +
      '  train=' + (ev.train_acc != null ? Number(ev.train_acc).toFixed(1) + '%' : '?') +
      '  val=' + (ev.val_acc != null ? Number(ev.val_acc).toFixed(1) + '%' : '?')
    );
  }

  function pad(v, n) {
    var s = (v == null ? '?' : String(v));
    while (s.length < n) s = ' ' + s;
    return s;
  }

  // =====================================================================
  // DOM helper
  // =====================================================================
  function el(name, attrs) {
    var node = document.createElementNS(SVG_NS, name);
    if (attrs) {
      for (var k in attrs) {
        if (!Object.prototype.hasOwnProperty.call(attrs, k)) continue;
        if (k === 'class') node.setAttribute('class', attrs[k]);
        else node.setAttribute(k, attrs[k]);
      }
    }
    return node;
  }

  function clamp(v, lo, hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
  }

  // =====================================================================
  // Public namespace
  // =====================================================================
  window.BrainReplay = {
    // Index page
    fetchTasks: fetchTasks,
    // Replay page
    init: init,
    loadTrace: loadTrace,
    initReplay: initReplay,
    // Transport (exposed for debugging / future test hooks)
    stepForward: stepForward,
    stepBackward: stepBackward,
    play: playTransport,
    pause: pause,
    scrubTo: scrubTo,
    setStep: setStep,
    layoutNeurons: layoutNeurons
  };
})();
