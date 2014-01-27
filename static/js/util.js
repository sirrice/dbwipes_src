
Array.prototype.clone = function() {return this.slice(0);};

function error(msg) {
  var div = d3.select("#messagebox").append("div")
    .classed('alert alert-danger',true)
    .text(msg);

  div.append("a")
    .classed('close', true)
    .attr({
      type: 'button',
      'aria-hidden': true
    })
    .html("&times;")
    .on('click', function() { div.remove() })
}

var onSelect = (function() {
  var cache = {};
  return function(rows, node, pstore, agg) {
    if (rows.length == 0) return;
    global_state.walkthrough.loading()
    var inputs = pstore.lookup(rows);
    global_state.highlighted_geoms[agg] = rows;
    global_state.highlighted_keys[agg] = inputs;
    global_state.walkthrough.render(d3.mouse(document.body))
  };
})()


function getYRange(data) {
  var miny = Infinity,
      maxy = -Infinity;
  _.each(labels.aggs, function(agg) {
    var vals = data.all(agg);
    miny = Math.min(miny, d3.min(vals));
    maxy = Math.max(maxy, d3.max(vals));
  })
  return [miny, maxy];

}

var setup_swap_events = function() {
  $("#swap-event-btn").click(swap_events);
};

var swap_events = function() {
  var drawEl = $(".drawing"),
      brushEl = $(".brush");

  if (brushEl.css("pointer-events") == "all") {
    brushEl.css("pointer-events", "none");
    drawEl.css("pointer-events", "all");
    $("#swap-event-btn").text("click when done");
  } else {
    drawEl.css("pointer-events", "none");
    brushEl.css("pointer-events", "all");
    global_state.drawn_path = render_drawing.path();
    $("#swap-event-btn").text("click to draw");
  } 
}

// assumes xs is sorted
var interpolate_path = function(xys, xs) {
  xys.sort(function(a,b) { return a[0] - b[0]; });
  var idx = -1,
      xyidx = 0,
      xy = xys[0],
      xyn = xys[1],
      ys = [],
      x = null;
  while (++idx < xs.length ){
    x = xs[idx];
    while (x >= xyn[0] && ++xyidx < xys.length) {
      xy = xyn;
      xyn = xys[xyidx]
    }
    if (xyidx >= xys.length) break;
    var ratio = (x-xy[0]) / (xyn[0]-xy[0]);
    ys.push(ratio * (xyn[1]-xy[1]) + xy[1])
  }
  return ys;
}


var render_drawing = (function(){
  var pathArr = null;
  var canvas = null;
  var canvasEl = null;
  var _plot = null;

  var f = function(plot) {
    _plot = plot;
    var svg = plot.svg;
    var panes = svg.selectAll(".data-pane");
    var w = +panes.attr('width'), 
        h = +panes.attr('height');

    canvas = panes.append("g").classed("drawing", true);
    canvasEl = canvas[0][0];

    canvas.append('rect')  
      .style({
        visibility:'hidden'
      })
      .attr('width', w)
      .attr('height', h)

    var line = d3.svg.line()
      .x(function(d) { return d[0] })
      .y(function(d) { return d[1] })
      .interpolate('basis');

    var isDrawing = false;
    pathArr = null;
    var path = canvas.append('path')
      .style({'stroke': 'black', 'stroke-width': '3px', 'fill': 'none'});
    var redraw = function() {
      path.attr('d', line(pathArr));
    };
      
      

    canvas
      .on('mousedown', function() {
        isDrawing = true;
        pathArr = [];
        redraw();
      })
      .on('mousemove', function() {
        if (!isDrawing) return;
        var xy = d3.mouse(canvasEl);
        if (pathArr.length > 1) {
          var diff = pathArr[pathArr.length-1][0] - pathArr[pathArr.length-2][0];
          if ((xy[0] - pathArr[pathArr.length-1][0])*diff > 0) {
            pathArr.push(d3.mouse(canvasEl));
            redraw();
          }
        } else if (pathArr.length == 1) {
          var diff = xy[0] - pathArr[pathArr.length-1][0];
          if (diff != 0) {
            pathArr.push(d3.mouse(canvasEl));
            redraw();
          }
        } else {
          pathArr.push(d3.mouse(canvasEl));
          redraw();
        }
      })
      .on('mouseup', function() {
        if (!isDrawing) return;
        if (pathArr.length <= 1) pathArr = null;
        isDrawing = false;
      })


  }

  f.path = function() { return pathArr; }
  f.status = function() { return canvas.style('pointer-events'); }
  f.disable = function() { return canvas.style('pointer-events', 'none') }
  f.enable = function() { return canvas.style('pointer-events', 'all') }
  f.plot = function() { return _plot; }

  return f;
})();

// data: list of objects
// labels: {
//  x: attr name for 'x'
//  aggs: attr name for 'y's
//  gbs: list of all group by attrs
//  id: attr name for 'id'
// }
var render_data = (function() {
  var _cache = {};
  return function(data, labels, cachekey) {
    data = fix_date_objects(data, labels);
    var thisyrange = getYRange(data);
    global_state.yrange[0] = Math.min(global_state.yrange[0], thisyrange[0]);
    global_state.yrange[1] = Math.max(global_state.yrange[1], thisyrange[1]);
    thisyrange = _.clone(global_state.yrange);

    if (cachekey != null && !(cachekey === undefined)) {
      if(cachekey in _cache) {
        var o = _cache[cachekey];
        if (o.lim[0] != thisyrange[0] || o.lim[1] != thisyrange[1]) {
          delete _cache[cachekey];
        } else {
          $("#aggplot").empty().append(o.svg[0][0]);
          render_drawing(o.plot);
          return;
        }
      }
    }

    var spec = {
      on: { },
      opts: {
        uri: "http://localhost:8881",
w: 900
      },
      data: data,
      layers: [],
      aes: {
        x: labels.x,//"{new Date("+ labels.x +")}",
        z: 1
      }, 
      scales: {
        r: { range: [0, 20] },
        y: { lim: [thisyrange[0], thisyrange[1]] },
        color: "identity"
      },
      debug: {
        'gg.wf.std': 5
      }
    }

    var cscale = d3.scale.category10().domain(_.range(labels.aggs.length));
    _.each(labels.aggs, function(agg, idx) {
      var layer = {
        geom: 'point',
        aes: {
          y: agg,
          color: cscale(idx)
        }
      };
      spec.layers.push(layer);
      spec.on['brushend-layer-'+idx] = function(rows, node, pstore) {
        onSelect(rows, node, pstore, agg);
      }
    });

    var plot = gg(spec);
    var span = d3.select("body").append("span").style('display', 'none');
    plot.render(span);//d3.select('#aggplot').append("span"));
    plot.on("done", function() {
      if (cachekey != null) {
        _cache[cachekey] = {
          lim: thisyrange,
          svg: plot.svg,
          plot: plot
        };
      }

      $("#aggplot").empty().append(plot.svg[0][0]);
      render_drawing(plot);
    });

    return plot;
  }
})();


function render_raw_table(data) {
	var keys = d3.keys(data[0]).filter(function(x){return x != 'x'}).slice(0, 20);
	keys = ['x'].concat(keys);

	var table = d3.select('#rawdata').append("table")
			.attr('class', 'table-striped table table-bordered  table-condensed	');

	var header = table.append('thead').append("tr").selectAll('th')
			.data(keys)
		.enter().append('th')
			.text(function(k) {return k == 'x'? 'Groupby id' : k;});

	var trs = table.append('tbody').selectAll('tr')
			.data(data)
		.enter().append('tr');

	keys.forEach(function(key) {
		trs.append('td')
			.text(function(d) {return d[key];});
	})			
	
}





// replaces ISOFORMAT strings in data with date objects
// modifies arguments
// XXX: removed labels argument. assume global labels vaiable never changes
function fix_date_objects(data){ //, labels) {
	if (!labels['dt'] || labels['dt'].length == 0)
		return data;

	var dt_labels = labels['dt'];
  data = data.project({
    alias: dt_labels,
    cols: dt_labels,
    f: function() {
      var ret = {};
      _.each(arguments, function(str, idx) {
        ret[dt_labels[idx]] = new Date(str);
      })
      return ret;
    }
  })
  return data;
}



/*****************
*
* Click handlers
*
*****************/




function update_query(sql) {
  var rest = (sql.split(/(SELECT|FROM|WHERE|GROUP BY|ORDER BY)/));
  rest = _.filter(rest, function(a) { return a.length > 0 } );
  rest = rest.map(function(a) { return a.trim() });
  var pretty = [];
  _.times(rest.length/2, function(idx) {
    pretty.push( rest[idx*2] + " " + rest[idx*2+1])
  })
  pretty = pretty.join('\n')


	$("#query, #errbox_query").text(pretty).val(pretty);
	global_state.query = pretty;
}


//
// Retrieve the aggregate query results for a given "dice" of the results
// Caches results
// Most recent callback function is executed when response arrives
//
//
var get_aggdata = (function(){
	var _loading = {};
	var _cache = {};
	var _funcs = {};
	return function(query, idx, where, cb_func) {
		cb_func = cb_func || function(arg){};
		_funcs[idx] = cb_func;

		if (!_cache[idx]) {


			if (!_loading[idx]) {
				_loading[idx] = true;
				var params = {'query' : query, 'filter' : where, 'db' : global_state.db}
				$.post('/json/query/', params, function(resp) {
          resp.data = gg.data.fromArray(resp.data);
					resp.data = fix_date_objects(resp.data)//;, resp.labels);
          var thisyrange = getYRange(resp.data);
          global_state.yrange[0] = Math.min(global_state.yrange[0], thisyrange[0]);
          global_state.yrange[1] = Math.max(global_state.yrange[1], thisyrange[1]);

					_cache[idx] = resp;
					_funcs[idx](resp);
				}, 'json')
			}
			return null;


		} else {


			_funcs[idx](_cache[idx]);


		}
		return _cache[idx];
	}
})();


function render_schema(selector,schema) {
	$(selector).empty();
	var fsdiv = d3.select(selector);
	var container = fsdiv.append('div').attr('class', 'row');
	container.append('div').attr('class', 'span2').html('&nbsp;');
	var tabcont = container.append('div').attr('class', 'span8')
	container.append('div').attr('class', 'span2')
	var tables = tabcont.selectAll("table")
			.data(d3.entries(schema))
		.enter().append("table")
			.attr('class', 'table-striped table table-bordered  table-condensed')

	var tbodys = tables.append('tbody');

	
	var ths = tbodys.data(d3.entries(schema)).append('tr').append('th')
			.attr('colspan', '2')
			.text(function(d) {return d.key})
			.style("text-align", "center")

	var trs = tbodys.selectAll('tr')
			.data(function(d) { console.log(['trs', d]); return d.value})
		.enter().append("tr")

	var tds = trs.selectAll("td")
			.data(function(d,i) {return d})
		.enter().append('td')
			.attr('width', '50%')
			.text(String)
			.style("font-size", "smaller");						
}




var onScorpionSubmit = function() {

	if (global_state.n_bad_keys() == 0) {
		error("Please select 1+ points to debug!")
		return false;
	}
  var c = +$("#c").val();
  if (!_.isFinite(+c) ) {
    error("c parameter is not a finite number: " + c);
  }
  $("#c").val(+c);


  
  if (global_state.drawn_path != null) {
    var n = 0;
    var labelattr = null;
    _.each(labels.aggs, function(yattr) {
      if (yattr in global_state.bad_keys && 
          global_state.bad_keys[yattr].length > 0) {
        n++;
        labelattr = yattr;
      }
    });
    if (n > 1) {
      error("When using drawing mode, only select one aggregate type as bad");
      return false;
    }

    var path = global_state.drawn_path;
    var bad_geoms = global_state.bad_geoms[labelattr];
    var bad_keys = global_state.bad_keys[labelattr];
    if (bad_keys.length != bad_geoms.length) {
      error("can't deal with aggregation in renderer.  badkeys != badgeoms");
      console.log(bad_keys);
      console.log(bad_geoms);
      return false;
    }
    var geomkeys = _.zip(bad_geoms, bad_keys);
    geomkeys.sort(function(a, b) { return a[0].get('x') - b[0].get('x') });
    bad_geoms = [];
    bad_keys = [];
    _.each(geomkeys, function(pair) {
      bad_geoms.push(pair[0]);
      bad_keys.push(pair[1]);
    });
    var xs = bad_geoms.map(function(r){return r.get('x')});
    var ys = interpolate_path(path, xs);
  

    // XXX huge hack!
    var plot = global_state.plot;
    var pt = plot.workflow.sinks()[0].inputs[0];
    var set = pt.right().any('scales');
    var yscale = set.get('y').clone()
    var yr = yscale.range();
    yscale.range([yr[1], yr[0]]);
    var erreq = {};
    erreq[labelattr] = _.map(ys, yscale.invert.bind(yscale));


    // ok setup all of the form inputs
    $("#errtype_1").attr('checked', true);
    var form_bad_keys = {};
    form_bad_keys[labelattr] = bad_keys.map(function(row) { 
      return row.get(labels.x);
    });
    var form_good_keys = {};

    global_state.attrs = [];
    $(".errattrs").get().forEach(function(d) {
      if(true || $(d).attr("checked")) {
        global_state.attrs.push($(d).val());
      }
    })

    $("#errbox_attrs").val(JSON.stringify(global_state.attrs));
    $("#errbox_errids").val(JSON.stringify(global_state.bad_tuple_ids));
    $("#erreq").val(JSON.stringify(erreq));
    $("#errbox_val").val(JSON.stringify(form_bad_keys));
    $("#errbox_goodkeys").val(JSON.stringify(form_good_keys));
    $("#errbox_db").val(global_state.db);

    if (erreq[labelattr].length != form_bad_keys[labelattr].length) {
      error("please draw a line across al of the outlier points");
      return false;
    }




    return true;
  }


  var form_bad_keys = {};
  var form_good_keys = {};


  _.each(labels.aggs, function(yattr) {
    try{
      var errtype = false;
      if (!errtype || errtype === undefined) {
        // assume only one agrgegate
        //var yattr = labels.aggs[0];
        if (!(yattr in global_state.bad_keys)) return;

        var bad_keys = global_state.bad_keys[yattr];
        var bad_vals = bad_keys.map(function(row) {
          return row.get(yattr);
        });

        var good_vals = null;
        var good_keys = null;
        if ((yattr in global_state.good_keys) && global_state.good_keys[yattr].length > 0) {
          good_keys = global_state.good_keys[yattr];
          good_vals = good_keys.map(function(row) { 
            return row.get(yattr);
          });
        } else {
          var bad_ids = bad_keys.map(function(row) { return row.id; });
          good_keys = data.filter(function(row) {
            return !_.contains(bad_ids, row.id);
            })
          
          good_vals = good_keys.map(function(row) {
            return row.get(yattr);
          })
          good_keys = _.sample(good_keys.all(), 10)
        }
      }

      console.log(d3.mean(bad_vals));
      console.log(d3.mean(good_vals));
      var avgbad = d3.mean(bad_vals),
        avggood = d3.mean(good_vals);
      if (avgbad > avggood) {
        errtype = "2";
      } else if (avgbad < avggood) {
        errtype = "3";
      } else {
        errtype = "0";
      }
      console.log("setting errtype to " + errtype)
      $("#errtype_"+errtype).attr("checked", true)

      form_bad_keys[yattr] = bad_keys.map(function(row) { 
        return row.get(labels.x);
      });
      form_good_keys[yattr] = good_keys.map(function(row) {
        return row.get(labels.x);
      });

      
    } catch(exception) {
      console.log(exception.stack)
      return false;
    }
  })
  
  
  global_state.attrs = [];
  $(".errattrs").get().forEach(function(d) {
    if(true || $(d).attr("checked")) {
      global_state.attrs.push($(d).val());
    }
  })

	$("#errbox_attrs").val(JSON.stringify(global_state.attrs));
	$("#errbox_errids").val(JSON.stringify(global_state.bad_tuple_ids));
	$("#errbox_val").val(JSON.stringify(form_bad_keys));
	$("#errbox_goodkeys").val(JSON.stringify(form_good_keys));
	$("#errbox_db").val(global_state.db);
	return true;	
}

