
Array.prototype.clone = function() {return this.slice(0);};

function error(msg) {
	div = $("<div/>").attr({'class':"alert alert-error"})
	a = $("<a/>").addClass('close').attr('data-dismiss', 'alert').text('x')
	div.append(a).text(msg);
	$("#messagebox").append(div);
}

var onSelect = function(rows, node, pstore) {
  var idstr = rows.map(function(row) { return row.id }).join(" ");
  if (idstr in cache) return cache[idstr];
  var inputs = pstore.lookup(rows);
  var ids = _.map(inputs, function(row) {
    return row.get(labels.x);
  });
  cache[idstr] = inputs;
  global_state.highlighted_keys = inputs;
  global_state.walkthrough.render()
};


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
          $("#aggplot").empty().append(_cache[cachekey].svg);
          return;
        }
      }
    }

    var spec = {
      on: {
        select: function(rows, node, pstore) {
          onSelect(rows, node, pstore);
        }
      },
      opts: {
        uri: "http://localhost:8881"
      },
      data: data,
      layers: [],
      aes: {
        x: labels.x,//"{new Date("+ labels.x +")}",
        z: 1
      }, 
      scales: {
        r: { range: [0, 20] },
        y: { lim: [thisyrange[0], thisyrange[1]] }
      }
    }

    _.each(labels.aggs, function(agg) {
      var layer = {
        geom: {
          type: 'point'
        },
        aes: {
          y: agg
        }
      };
      spec.layers.push(layer);
    });

    var plot = gg(spec);
    $("#aggplot").empty();
    plot.render(d3.select('#aggplot').append("span"));
    plot.on("done", function() {
      if (cachekey != null) {
        _cache[cachekey] = {
          lim: thisyrange,
          svg: $("#aggplot svg")
        };
      }
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
function fix_date_objects(data, labels) {
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
					resp.data = fix_date_objects(resp.data, resp.labels);
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

