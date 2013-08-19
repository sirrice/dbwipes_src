
Array.prototype.clone = function() {return this.slice(0);};
function clone(o) {
	var ret = {};
	d3.keys(o).map(function(key) {
		ret[key] = o[key].clone();
	})
	return ret;
}







function error(msg) {
	div = $("<div/>").attr({'class':"alert alert-error"})
	a = $("<a/>").addClass('close').attr('data-dismiss', 'alert').text('x')
	div.append(a).text(msg);
	$("#messagebox").append(div);
}


var Renderer = function(rootid, brushevent, opts) {
	opts = opts || {};
	console.log(['render_scatter opts', opts, opts.r||1.5])
	this.rootid = rootid;
	this.brushevent = brushevent;
	this.overlap = opts.overlap || -2;
	this.r = opts.r || 1.5;
	this.h = opts['h'] || 300;
	this.w = opts['w'] || 800;
	this.px = 40;
	this.py = 30;
	this.sx = (this.w-(2*this.px)) / this.w;
	this.sy = (this.h-(2*this.py)) / this.h;
	this.jsvg = null;
	this.jlegend = null;
	this.xlabeldiv = null;
	this.svg = null;
	this.circlecontainer = null;
	this.minx = this.maxx = this.miny = this.maxy = null;	

	this.on('brushevent', brushevent, this);
}

_.extend(Renderer.prototype, Backbone.Events, {

	update_opts: function(opts) {
		if (!opts) return;
		this.overlap = opts['overlap'] || this.overlap || -2;
		this.r = opts['r'] || this.r || 1.5;
		this.h = opts['h'] || this.h || 300;
		this.w = opts['w'] || this.w || 800;		
	},

	update_bounds: 	function (nx, mx, ny, my) {
		if (this.maxx == null || mx > this.maxx)// || this.maxx - mx > 0.6 * (this.maxx-this.minx)) 
			this.maxx = mx;
		if (this.maxy == null || my > this.maxy)// || this.maxy - my > 0.6 * (this.maxy-this.miny)) 
			this.maxy = my;
		if (this.minx == null || nx < this.minx)// || nx - this.minx > 0.6 * (mx-this.minx)) 
			this.minx = nx;
		if (this.miny == null || ny < this.miny)// || ny - this.miny > 0.6	 * (my-this.miny)) 
			this.miny = ny;
		console.log(['bounds',[nx, mx, ny, my], 
					 [this.minx, this.maxx, this.miny, this.maxy]]);		

	},


	summarize_scatter: function(data, labels, color) {
		//label.x, label.y, label.ps
		var h = this.h,
			w = this.w,
			r = this.r*2,
			overlap = this.overlap,
			minx = d3.min(data.map(function(d){return d[labels.x];})),
			maxx = d3.max(data.map(function(d){return d[labels.x];})),
			miny = d3.min(data.map(function(d){return d[labels.y];})),
			maxy = d3.max(data.map(function(d){return d[labels.y];})),
			maxptsx = w / Math.max(r-overlap, 1), // maximum number of points along x axis
			maxptsy = h / Math.max(r-overlap, 1), // maximum number of points along y axis
			dx = Math.max((maxx - minx) / maxptsx, 1),
			dy = Math.max((maxy - miny) / maxptsy, 1);
		
		var data_gb = d3.nest()
			.key(function(d) {	return Math.floor(d[labels.x] / dx) * dx;	})
			.key(function(d) {	return Math.floor(d[labels.y] / dy) * dy;	})
			.entries(data);


		// remember how many values there are
		var minpts = null, maxpts = null;
		data_gb.forEach(function( xgroup ) {
			xgroup.values.forEach( function( ygroup ) {
				var nvals = ygroup.values.length;	
//				console.log([data[0], labels.x, maxx, minx, maxptsx, dx, xgroup.key, ygroup.key]);
				ygroup['__label__'] = labels.y;
				ygroup['__x__'] = Number(xgroup.key);
				ygroup['__y__'] = Number(ygroup.key);
				minpts = (minpts == null || minpts > nvals)? nvals : minpts;
				maxpts = (maxpts == null || maxpts < nvals)? nvals : maxpts;
			})
		});
		minpts = (minpts == null)? 0 : minpts;
		maxpts = (maxpts == null)? 1 : maxpts;

		var mincolor = d3.rgb(color).brighter().hsl(),
			maxcolor = d3.hsl(color);
		maxcolor.s = 1;
		mincolor.s = 0.2;
		var cscale = d3.scale.linear().range([mincolor.toString(), maxcolor.toString()]).domain([minpts, maxpts]);
		cscale = function(d,i) {return color;};
		var rscale = d3.scale.linear().range([this.r, this.r+2.5]).domain([minpts, maxpts]);

		var bounds = {	'nx': Math.floor(minx/dx)*dx,
						'mx': Math.floor(maxx/dx)*dx,
						'ny': Math.floor(miny/dy)*dy,
						'my': Math.floor(maxy/dy)*dy,
						'dx': dx,
						'dy': dy };
		console.log(bounds);
		return {'data' : data_gb, 'cscale' : cscale, 'bounds' : bounds, 'rscale' : rscale}
	}, 


	add_axes: function(xscale, yscale, ylabels) {
//		xscale = d3.time.scale().domain(xscale.domain()).range([this.px, this.w-this.px]);
		yscale = d3.scale.linear().domain(yscale.domain()).range([this.h-this.py, this.py]);

		var xrules = this.svg.append('g').selectAll('.xlabel')
				.data(xscale.ticks(8))
			.enter().append('g')
				.attr('class', 'xlabel')
				.attr('transform', function(d) {console.log([xscale(d), d]);return 'translate('+xscale(d)+' 0)'})
		xrules.append('text')
				.attr('y', this.h)
				.text(xscale.tickFormat(8))
		this.svg.append('line')
		var yrules = this.svg.append('g').selectAll('.ylabel')
				.data(yscale.ticks(6))
			.enter().append('g')
				.attr('class', 'ylabel')
				.attr('transform', function(d) {return 'translate(0 '+yscale(d)+')'})
		yrules.append('text')
				.attr('x', 0)
				.text(String);
		yrules.append('line')
				.attr('x1', this.py)
				.attr('x2', this.w)
				.attr('stroke', '#ccc');		
		

	},

	is_date_scale: function(label, labels) {
		for (var i = 0; i < labels['dt'].length; i++) {
			if (label == labels['dt'][i]) 
				return true;
		}
		return false;
	},	



	// user should specify
	//  possible group by labels
	//  possible aggregate points 
	//  displayed aggregate points
	//  how to render an aggregated point (currently size based on #points in the region)
	//  current x axis label 
	//  current y axis label (either None to use)

	render_scatter: function (_data, _labels, opts) {
		if (!_labels || typeof(_labels) == 'undefined') {
			error("Did not get any data to render!")
			return;
		}

		$("#"+this.rootid).show();
		$("#rawdata").hide();

		try {
			this.update_opts(opts);
			this.jsvg = $("#"+this.rootid + " svg"),
			this.jlegend = $("#"+this.rootid+" .legend");
			this.xlabeldiv = $("#"+this.rootid+" .xlabel");		
			this.jsvg.empty(); this.jlegend.empty(); this.xlabeldiv.empty();
			var self = this;		
			console.log([this.rootid, this.jsvg, this.jlegend]);
			console.log(["_labels", _labels]);
			console.log(["_data", _data])

			var labels = _labels.aggs, 
				x_label = _labels.x,
				cscale = d3.scale.category10().domain(labels),
				nx = mx = ny = my = null,
				summaries = labels.map(function(label) {
					summary = self.summarize_scatter(_data, {'x':x_label, 'y':label}, cscale(label));
					nx = (nx == null || nx > summary.bounds.nx)? summary.bounds.nx : nx;
					mx = (mx == null || mx < summary.bounds.mx)? summary.bounds.mx : mx;
					ny = (ny == null || ny > summary.bounds.ny)? summary.bounds.ny : ny;
					my = (my == null || my < summary.bounds.my)? summary.bounds.my : my;
					return summary;
				});


			this.update_bounds(nx, mx, ny, my);
			console.log([this.minx, this.maxx]);

			var	ysc = ((1-((this.h-(this.py*2) - 5) / (this.h-(this.py*2)))) / 2.0) * (this.maxy - this.miny),
				xsc = ((1-((this.w-(this.px*3) - 10) / (this.w-(this.px*3)))) / 2.0) * (this.maxx - this.minx),
				ysc = isNaN(ysc)? 10 : ysc,
				xsc = isNaN(xsc)? 10 : xsc,
				yscale = d3.scale.linear().domain([this.miny - ysc, this.maxy + ysc]).range([this.h-this.py, this.py]),
				xscale = d3.scale.linear().domain([this.minx - xsc, this.maxx + xsc]).range([this.px, this.w-this.px]),		
				ex = [xscale(this.minx*(1-xsc)), xscale(this.maxx*(1+xsc))],
				ey = [yscale(this.miny*(1-ysc)), yscale(this.maxy*(1+ysc))];
			console.log(["scaling", ysc, xsc])



			var legend = d3.selectAll(this.jlegend.get()).selectAll('text')
					.data(labels)
				.enter().append('div')
					.style('float', 'left')
					.style('color', cscale)
					.text(String);		
			
			//
			// handle x-axis select options
			//
			var xaxisselect = this.xlabeldiv.append($("<select></select")).find("select");
			var xaxislabel = d3.selectAll(xaxisselect.get()).selectAll("option")
					.data(_labels.gbs)
				.enter().append("option")
					.attr("value", String)
					.text(String);
			xaxisselect.val(x_label);
			(function() {
				var selectedval = x_label;
				$("#"+self.rootid+" .xlabel select").change(function() {
					var val = $("#"+self.rootid+" .xlabel select").val();
					console.log(["selected option", selectedval, val])				
					if (val == selectedval) return;
					selectedval = val
					var newlabels = {"x" : val, "gbs" : _labels.gbs, "aggs" : _labels.aggs}

					self.render_scatter(_data, newlabels, opts);
				});
			})();

			this.svg = d3.selectAll(this.jsvg.get())
				.attr('width', this.w)
				.attr('height', this.h)
				.attr('class', 'g')
				.attr('id', 'svg_'+this.rootid);


			if (this.is_date_scale(_labels.x, _labels)) {
				console.log(['is date scale', _labels.x]);
				var timexscale = d3.time.scale()
					.domain([this.minx, this.maxx])
					.range([this.px, this.w-this.px]);
				this.add_axes(timexscale, yscale, _labels.gbs);
			} else {
				console.log(['not date scale', _labels.x]);				
				this.add_axes(xscale, yscale, _labels.gbs);
			}
						
			this.circlecontainer = this.svg.append('g')
				.attr("class", "circlecontainer")
				.attr('width', ex[1]-ex[0])
				.attr('height', ey[0]-ey[1])
				.attr("x", ex[0])
				.attr("y", ey[1]);


			function add_circle (summary, idx) {
				var label = labels[idx];
				var color = cscale(label);

				var cellcont = self.circlecontainer.append('g')
						.attr("id", self.rootid+"_cells_"+idx)
						.attr("x", ex[0])
						.attr("width", ex[1] - ex[0])
						.attr("y", ey[1])
						.attr("height",  Math.abs(ey[1] - ey[0]))
						.attr('class', "circleplot");
				
				var cells = cellcont.selectAll('g')
						.data(summary.data)
					.enter().append('g')

				cells.selectAll('circle')
						.data(function(d) { return d.values;})
					.enter().append('circle')
						.attr('cy', function(d) { return yscale(d.key)})
						.attr('cx', function(d) { return xscale(d['__x__'])})
						.attr('r', function(d){return summary.rscale(d.values.length)})
            .attr('opacity', 0.8)
						.attr('fill', function(d) { d['__color__'] = summary.cscale(d.values.length); return d.__color__; })
						.attr('label', label);
			}
			summaries.forEach(add_circle);

			///////////////////////////////////
			// BRUSHES!
			///////////////////////////////////
			var selectedobjs = {};			
			function brushf() {
				var extents = brush.extent();
				labels.forEach(function(l) {selectedobjs[l] = [];})
				console.log([self.svg.attr('id'), extents]);
				self.svg.selectAll('circle')
					.attr('fill', function(d, i) {
						var c = d3.select(this),
								x = c.data()[0]['__x__'],
								y = c.data()[0]['__y__'];
						var valid = extents[0][0] <= x &&
									extents[1][0] >= x &&
									extents[0][1] <= y &&
									extents[1][1] >= y;
						if (valid) {
							var list = selectedobjs[c.attr('label')];
							if (typeof(list) != 'undefined')
								list.push.apply(list, c.data()[0].values);
						}
						return valid ? 'black' : c.data()[0]['__color__'];
				})
				


				if (self.brushevent !== null && typeof(self.brushevent) != "undefined")
					this.trigger('brushevent', selectedobjs, _labels, extents);					
	//				self.brushevent(selectedobjs, _labels);			
			}



			var brush = d3.svg.brush()
				.on('brush', $.proxy(brushf, this))
				.x(xscale)
				.y(yscale);

			this.svg.append('g')
					.attr('class', 'brush')
					.call(brush);

		} catch(err) {
			console.log(['error', err])
			$("#"+this.rootid).hide();
			$("#rawdata").show();
			render_raw_table(_data);
		}

	}
})


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





/*****************
*
* Brushes 
*
*****************/

function aggbrush(selectedobjs, _labels) {
	d3.keys(selectedobjs).forEach(function(k) {
		selectedobjs[k] = selectedobjs[k].map(function(d) {return d[_labels.x]});
	});

	nerrpts = d3.sum(d3.values(selectedobjs).map(function(d) {return d.length}));
	//$("#errbox_val").val(JSON.stringify(selectedobjs));
	$("#tab-aggplot .err_npts").text(nerrpts);	
	global_state.highlighted_keys = selectedobjs;
	//global_state.selected_keys = selectedobjs;
}

function tupbrush(selectedobjs, _labels) {
	//selectedxs = {}	
	selectedids = {}
	d3.keys(selectedobjs).forEach(function(k) {
		selectedids[k] = selectedobjs[k].map(function(d) {return d[_labels.id]});
	});

	nerrpts = d3.sum(d3.values(selectedobjs).map(function(d) {return d.length}));
	$("#tab-tupplot .err_npts").text("("+nerrpts+" pts)");	
	global_state.highlighted_tuple_ids = selectedids;
	//global_state.bad_tuple_ids = selectedids;
}




// replaces ISOFORMAT strings in data with date objects
// modifies arguments
function fix_date_objects(data, labels) {

	if (!labels['dt'] || labels['dt'].length == 0)
		return;

	var dt_labels = labels['dt'];
	for (var i = 0; i < data.length; i++) {
		var pt = data[i];
		for (var j = 0; j < dt_labels.length; j++) {
			pt[dt_labels[j]] = new Date(pt[dt_labels[j]]);
		}
	}
}



/*****************
*
* Click handlers
*
*****************/
function update_aggplot() {
	var params = {
		'query' : global_state.query,
		'filter' : global_state.filter,
		'db' : global_state.db
	}
	$.post('/json/query/', params, function(resp) {
		fix_date_objects(resp.data, resp.labels);
		renderagg.render_scatter(resp.data, resp.labels);
	}, 'json')
}


// Sends the _highighted_ points to the server
function update_tupleplot() {
	if (global_state.n_selected_keys() == 0) {
//		error("Please select some points so we know where to zoom in to!");
		return;
	}
    
    var params = {
    	'data' : JSON.stringify( global_state.highlighted_keys ),
    	'query' : global_state.query,
    	'filter' : global_state.filter,
    	'clauseid' : global_state.clauseid
    }
    $("#tab-tupplot .heading small").text("loading...");		
	get_tuples(params, function(resp) {
		$("#tab-tupplot .heading small").text("rendering...");		
		fix_date_objects(resp.data, resp.labels);
		rendertup.render_scatter(resp.data, resp.labels);
		//render_raw_table(resp);
		$("#tab-tupplot .heading small").empty();
	});
}

function update_query(sql) {
	$("#query, #errbox_query").text(sql);	
	global_state.query = sql;
}

//
// Retrieve the raw tuples for a given "dice" of the results
// @param params = {'data' : [group by keys], 'query' : '', 'filter' : 'predicate that user clicks'}
// @param cb_func: function to call with response data
//
var get_tuples = (function(){ 
	var _cache = {};
	var _loading = {};
	var _funcs = {}
	return function(params, cb_func) {

		var clauseid = params.clauseid + "__" + params.data;
		_funcs[clauseid] = cb_func;

		if (typeof(_cache[clauseid]) == 'undefined'){

			if (!_loading[clauseid]) {
				_loading[clauseid] = true;
				params.db = global_state.db;
				$.post('/json/filterq/', params, function(resp) {
					_cache[clauseid] = resp;
					_funcs[clauseid](resp);
					
				}, 'json')
			} 			

		} else {

			_funcs[clauseid](_cache[clauseid]);

		}
		return _cache[clauseid];
	} 
})();


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
					fix_date_objects(resp.data, resp.labels);
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

