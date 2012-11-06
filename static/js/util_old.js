

function gen_render_scatter(rootid, brushevent) {

	var maxy = null;
	var maxx = null;
	var miny = 0;
	var minx = 0;

	function render_me(data) {
		if (Object.keys(data).length == 0) {
			error("Data does not contain Keys.");
			return;
		}

		var jsvg = $("#"+rootid + " svg"), jlegend = $("#"+rootid+" .legend");

		var labels = d3.keys(data[0]).filter(function(el){return el != 'x' && el != 'id'});
		var xs = data.map(function(el){return el.x});
		var w = jsvg.parent().parent().width(),
			h = 300,
			lh = 10,
			yaxiswidth = 50,
			xaxisheight = 20,
			nx = d3.min(xs),
			ny = d3.min(labels, function(l) {
				return d3.min(data.map(function(d) {return d[l]}));
			}),
			mx = d3.max(xs),
			my = d3.max(labels, function(l) { 
				return d3.max(data.map(function(d){return d[l]}));
			});


		var charth = h - lh - 20,
			chartw = w - yaxiswidth,
			chartyoff = lh,
			chartxoff = yaxiswidth,
			chartyscale = charth / h,
			chartxscale = chartw / w;
					
		if (maxx == null || mx > maxx || maxx - mx > 0.3 * (maxx-minx)) {
			maxx = mx;
		} else {
			mx = maxx;
		}
		if (maxy == null || my > maxy || maxy - my > 0.3 * (maxy-miny)) {
			maxy = my;
		} else {
			my = maxy;
		}
		if (nx < minx || nx - minx > 0.3 * (mx-minx)) {
			minx = nx;
		} else {
			nx = minx;
		}
		if (ny < miny || ny - miny > 0.3 * (my-miny)) {
			miny = ny;
		} else {
			ny = miny;
		}
		var padx = (mx - nx) * 0.1, pady = (my - ny) * 0.1;
		mx = mx + padx;
		my = my + pady;
		nx = nx - padx;
		ny = ny - pady;

		var xscale = d3.scale.linear().range([yaxiswidth, w]).domain([nx, mx]),
  		  	yscale = d3.scale.linear().range([charth-20, 0]).domain([ny, my]),
			lscale = d3.scale.linear().range([20, lh]).domain([0, labels.length]);

		var cscale = d3.scale.category10().domain(labels);
	//			yscale = d3.scale.linear().range([0, h]).domain([0, d3.max(y)]);
		
		jsvg.empty();  jlegend.empty();

		var svg = d3.selectAll(jsvg.get())
				.attr('width', w)
				.attr('height', h)
				.attr('class', 'g');

		// legend
		var legend = d3.selectAll(jlegend.get()).selectAll('text')
		//var legend = svg.append('g').selectAll('text')
				.data(labels)
			.enter().append('div')
				.style('float', 'left')
				.style('color', cscale)
				.text(String);
						
		
		// contains plot, xaxis, and yaxis
		var chart = svg.append('g')
				.attr('transform', 'translate(0 '+lh+')')
				.attr('width', w)
				.attr('height', h-lh);
		
		var xscale = d3.scale.linear().range([yaxiswidth, w]).domain([nx, mx])
		var xaxis = chart.append('g').selectAll('.xlabel')
				.data(xscale.ticks(10))
			.enter().append('g')
				.attr('class', 'xaxs')
				.attr('transform', function(d) {return 'translate('+xscale(d)+' 0)'})
			.append('text')
				.attr('y', h-lh)
				.text(String)

		// contains plot and xaxis
		var plotandxcontainer = chart.append('g')
				.attr('width', '100%')
				.attr('height', h-lh-xaxisheight);

		
		var yscale = d3.scale.linear().range([h-lh-xaxisheight, 0]).domain([ny, my])
		var yaxis = plotandxcontainer = chart.append('g');
		var rules = yaxis.selectAll('.rule')
				.data(yscale.ticks(5))
			.enter().append('g')
				.attr('class', 'rule')
				.attr("transform", function(d) { return  "translate(0 " + yscale(d) + ")"; });
		rules.append('text')
			    .attr("x", 0)
	    		.attr("dy", "0.35em")
				.text(String);
		rules.append('line')
				.attr('x1', yaxiswidth)
				.attr('x2', w)
				.attr('stroke', '#ccc');
		

		yscale = d3.scale.linear().range([h-lh-xaxisheight, 0]).domain([ny, my]);
		xscale = d3.scale.linear().range([0, w-yaxiswidth]).domain([nx, mx]);
		var plotcontainer = plotandxcontainer.append('g')
				.attr('transform', 'translate('+yaxiswidth+' 0)')
				.attr('width', w - yaxiswidth)
				.attr('height', h-lh-xaxisheight);





		function add_circles(container, fixedclass) {
			labels.forEach(function(label) {
				//newdata = data.map(function(d) {
				//	return {'x' : d.x, 'y' : d[label], 'id' : d['id']};
				//});
				circles = container.append('g').selectAll('circle')
						.data(data)
					.enter().append('circle')
						.attr('cx', function(d){ return xscale(d.x)})
						.attr('cy', function(d){ return yscale(d[label])})
						.attr('r', 1)		
						.attr('stroke-width', 0)
						.attr('idx', function(d, i) {return i;})
						.attr('label', label)

				if (fixedclass === null || typeof fixedclass === 'undefined') {
					circles.attr('fill', cscale(label));
				} else {
					circles.attr('class', fixedclass);
				}

			})
		}


		var background = plotcontainer.append('g')
				.attr('width', w - yaxiswidth)
				.attr('height', h-lh-xaxisheight);
		var foreground = plotcontainer.append('g')
				.attr('width', w - yaxiswidth)
				.attr('height', h-lh-xaxisheight);
		add_circles(background, 'bgcircle');
		add_circles(foreground);




		/*
		* BRUSHES!
		*/
		selectedobjs = {};			
		function brushf() {
			extents = brush.extent();
			labels.forEach(function(l) {selectedobjs[l] = [];})
			foreground.selectAll('circle')
				.style('display', function(d, i) {
					var c = d3.select(this),
							x = c.data()[0].x,
							y = c.data()[0][c.attr('label')];
					var valid = extents[0][0] <= x &&
						extents[1][0] >= x &&
						extents[0][1] <= y &&
						extents[1][1] >= y;
					if (valid) {
						selectedobjs[c.attr('label')].push(c.data()[0]);
					}
					// if valid, remove the foreground
					return valid ? 'none' : null;
			})
			
			if (brushevent !== null && typeof(brushevent) != "undefined")
				brushevent(selectedobjs);
		}

		var brush = d3.svg.brush().x(xscale).y(yscale).on('brush', brushf);
		plotcontainer.append('g')
				.attr('width', '100%')
				.attr('height', '100%')
				.attr('class', 'brush')
				.call(brush);

	}
	return render_me;
}